"""Ray-cast based fitness evaluation for the LiDAR layout GA.

This module replaces the analytic mock in ``evaluate_fitness_mock.py`` with a
physically grounded coverage estimate. For every sensor in an individual we:

  1. generate its local ray bundle from its FOV / resolution spec (cached),
  2. rotate + translate the bundle onto the sensor's mounting node,
  3. cast every ray against the robot chassis (``open3d.t.geometry`` tensor
     raycaster) to find the self-occlusion distance ``t_hit``,
  4. analytically intersect each ray with two evaluation surfaces -- the ground
     plane ``Z = 0`` (a disk of radius ``R_max``) and a surrounding cylinder of
     radius ``R_max`` -- and keep a hit only if it occurs *before* the chassis
     blocks the ray and within the sensor's range,
  5. rasterise the surviving hits into two occupancy grids (S_gnd, S_cyl).

Both evaluation surfaces share a single maximum evaluation radius ``R_max``:

  * **S_gnd** -- the ground is an annular disk discretised in *polar* coordinates
    ``(r, theta)``, extending outward from the robot's structural footprint
    (inner radius) to ``R_max`` (outer radius).
  * **S_cyl** -- the cylindrical wall stands at exactly ``R_max`` (the rim of the
    ground disk), discretised in cylindrical coordinates ``(z, theta)`` from the
    ground ``z = 0`` up to a maximum height.

Fitness is a normalised, weighted blend of grid coverage and financial cost.

Coordinate convention (matches ``utils/visualization.py`` and the mock):
    x = forward, y = left, z = up; positive pitch tilts the sensor UP.

Usage (notebook cell)::

    import custom_toolbox.evaluate.evaluate_fitness_raycast as evaluate
    evaluator = evaluate.CoverageEvaluator()
    toolbox.register("evaluate", evaluator.evaluate_individual)
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from config.graph import MOUNTING_GRAPH
from config.params import Individual
from config.sensors import Sensor, SensorType

# Single source of truth for the chassis mesh location.
from generate_mounting_graph import CHASSIS_MESH_PATH

# Numerical epsilon used to reject grazing / zero-length parametric distances.
_EPS = 1e-9


class CoverageEvaluator:
    """Stateful, reusable coverage evaluator for DEAP.

    The expensive setup -- loading the chassis mesh and building the raycasting
    scene -- happens once in ``__init__``. Local ray templates are cached per
    sensor type, so the per-individual cost is dominated by a single batched
    ``cast_rays`` call plus vectorised NumPy intersection math.
    """

    def __init__(
        self,
        mesh_path: str = str(CHASSIS_MESH_PATH),
        *,
        # --- Maximum evaluation radius R_max (shared by both surfaces) ---
        max_radius_m: float = 10.0,
        # --- Ground grid: annular disk in polar coords (r, theta) at Z = 0 ---
        ground_r_min_m: float | None = 0.0,  # inner radius; None => auto from mesh
        ground_r_res_m: float = 0.1,
        ground_azimuth_bins: int = 360,
        # --- Cylinder grid (height Z, azimuth theta at radius R_max) ---
        cyl_z_min_m: float = 0.0,
        cyl_z_max_m: float = 4.0,
        cyl_z_res_m: float = 0.1,
        cyl_azimuth_bins: int = 360,
        # --- Fitness blend ---
        w_cov: float = 0.8,
        w_cost: float = 0.2,
        max_budget: float = 5000.0,
    ) -> None:
        # ------------------------------------------------------------------
        # 1. Build the raycasting scene ONCE.
        # ------------------------------------------------------------------
        legacy_mesh = o3d.io.read_triangle_mesh(mesh_path)
        if len(legacy_mesh.triangles) == 0:
            raise ValueError(f"Loaded chassis mesh has no triangles: {mesh_path}")
        legacy_mesh.compute_vertex_normals()

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh))

        # Maximum evaluation radius: the outer rim of the ground disk and the
        # radius at which the cylindrical wall stands.
        self.max_radius = float(max_radius_m)

        # ------------------------------------------------------------------
        # 2. Ground grid geometry (annular disk in polar coords r, theta).
        #
        # The disk spans [r_min, R_max], where r_min is the robot's structural
        # footprint -- rays cannot reach the ground beneath the chassis, so the
        # interior is excluded from coverage scoring. When not supplied, r_min is
        # derived from the chassis mesh's maximum extent in the X/Y plane.
        # ------------------------------------------------------------------
        if ground_r_min_m is None:
            verts = np.asarray(legacy_mesh.vertices, dtype=float)
            self.ground_r_min = (
                float(np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2).max())
                if len(verts)
                else 0.0
            )
        else:
            self.ground_r_min = float(ground_r_min_m)
        # Guard against a footprint that swallows the whole evaluation disk.
        self.ground_r_min = min(self.ground_r_min, self.max_radius - _EPS)

        self.ground_r_res = float(ground_r_res_m)
        self.ground_n_r = max(
            1, int(round((self.max_radius - self.ground_r_min) / ground_r_res_m))
        )
        self.ground_n_az = int(ground_azimuth_bins)
        self.ground_dtheta = 2.0 * np.pi / self.ground_n_az

        # ------------------------------------------------------------------
        # 3. Cylinder grid geometry (vertical wall at radius R_max).
        # ------------------------------------------------------------------
        self.cyl_radius = self.max_radius
        self.cyl_z_min = float(cyl_z_min_m)
        self.cyl_z_res = float(cyl_z_res_m)
        self.cyl_nz = int(round((cyl_z_max_m - cyl_z_min_m) / cyl_z_res_m))
        self.cyl_n_az = int(cyl_azimuth_bins)
        self.cyl_dtheta = 2.0 * np.pi / self.cyl_n_az

        self._ground_cells_total = self.ground_n_r * self.ground_n_az
        self._cyl_cells_total = self.cyl_nz * self.cyl_n_az
        self._cells_total = self._ground_cells_total + self._cyl_cells_total

        # ------------------------------------------------------------------
        # 4. Fitness weights.
        # ------------------------------------------------------------------
        self.w_cov = float(w_cov)
        self.w_cost = float(w_cost)
        self.max_budget = float(max_budget)

        # ------------------------------------------------------------------
        # 5. Per-sensor-type local ray cache.
        # ------------------------------------------------------------------
        self._local_ray_cache: dict[SensorType, np.ndarray] = {}

        # Most recent grids, kept for debugging / visualisation.
        self.last_ground_grid: np.ndarray | None = None
        self.last_cyl_grid: np.ndarray | None = None

    # ======================================================================
    # Step 2: Sensor ray generation (local sensor frame)
    # ======================================================================
    def _generate_local_rays(self, sensor: Sensor) -> np.ndarray:
        """Return cached unit direction vectors ``(N, 3)`` for a sensor type.

        Spherical -> Cartesian with x=forward, y=left, z=up. Computed once per
        ``SensorType`` and reused for every gene of that type.
        """
        cached = self._local_ray_cache.get(sensor.sensor_type)
        if cached is not None:
            return cached

        num_h = max(1, int(round(sensor.fov_horizontal_deg / sensor.horizontal_res_deg)))
        num_v = max(1, int(sensor.vertical_channels))

        # Full 360 deg FOV must not duplicate the seam ray (endpoint=False);
        # a limited FOV should include both edges (endpoint=True).
        is_full_circle = sensor.fov_horizontal_deg >= 360.0 - 1e-6
        h_angles = np.deg2rad(
            np.linspace(
                -sensor.fov_horizontal_deg / 2.0,
                sensor.fov_horizontal_deg / 2.0,
                num_h,
                endpoint=not is_full_circle,
            )
        )

        if num_v == 1:
            v_angles = np.array([0.0])
        else:
            v_angles = np.deg2rad(
                np.linspace(
                    -sensor.fov_vertical_deg / 2.0,
                    sensor.fov_vertical_deg / 2.0,
                    num_v,
                )
            )

        H, V = np.meshgrid(h_angles, v_angles)
        H = H.ravel()
        V = V.ravel()

        dirs = np.stack(
            (np.cos(V) * np.cos(H), np.cos(V) * np.sin(H), np.sin(V)),
            axis=-1,
        ).astype(np.float32)

        self._local_ray_cache[sensor.sensor_type] = dirs
        return dirs

    # ======================================================================
    # Step 3: Gene transformation (local -> world frame)
    # ======================================================================
    @staticmethod
    def _rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
        """Body->world rotation, repo convention (positive pitch = UP).

        Equivalent to ``Rz(yaw) @ Ry(-pitch) @ Rx(roll)``; the negated pitch
        makes a forward ray map to ``z = +sin(pitch)``, matching
        ``visualization.py`` and the mock's ``OPTIMAL_PITCH = -10`` (down).
        """
        return Rotation.from_euler(
            "ZYX", [yaw, -pitch, roll], degrees=True
        ).as_matrix()

    def _transform_rays(
        self,
        local_rays: np.ndarray,
        node_xyz: np.ndarray,
        pitch: float,
        roll: float,
        yaw: float,
    ) -> np.ndarray:
        """Rotate ``local_rays`` and attach the mounting origin.

        Returns an ``(N, 6)`` array laid out as
        ``[ox, oy, oz, dx, dy, dz]`` ready for Open3D.
        """
        R = self._rotation_matrix(pitch, roll, yaw).astype(np.float32)
        world_dirs = local_rays @ R.T  # (N, 3), float32

        n = world_dirs.shape[0]
        rays6 = np.empty((n, 6), dtype=np.float32)
        rays6[:, 0:3] = node_xyz.astype(np.float32)
        rays6[:, 3:6] = world_dirs.astype(np.float32)
        return rays6

    # ======================================================================
    # Step 4: Raycasting & self-occlusion
    # ======================================================================
    def _cast(self, rays6: np.ndarray) -> np.ndarray:
        """Cast ``(M, 6)`` rays, returning the ``(M,)`` hit distances.

        ``t_hit`` is +inf for rays that never strike the chassis.
        """
        rays_t = o3d.core.Tensor(rays6, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays_t)
        return ans["t_hit"].numpy()

    # ======================================================================
    # Step 5: Discretisation & scoring (per evaluation surface)
    # ======================================================================
    def _ground_intersection(
        self, O: np.ndarray, D: np.ndarray, t_hit: np.ndarray, ranges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ground-plane (Z = 0) ray parameters.

        Returns ``(valid, t)`` where ``t`` is the parametric hit distance for
        every ray (``nan``/garbage where invalid) and ``valid`` flags the rays
        that strike the ground before the chassis occludes them and within range.
        """
        dz = D[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -O[:, 2] / dz
        valid = (dz != 0.0) & (t > _EPS) & (t < t_hit) & (t < ranges)
        return valid, t

    def _cylinder_intersection(
        self, O: np.ndarray, D: np.ndarray, t_hit: np.ndarray, ranges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cylinder-wall (radius R_max) ray parameters, nearest positive root.

        Returns ``(valid, t)`` analogous to :meth:`_ground_intersection`. ``t``
        is ``+inf`` where the ray has no positive root or the quadratic is
        degenerate, so it is always safe to compare.
        """
        ox, oy = O[:, 0], O[:, 1]
        dx, dy = D[:, 0], D[:, 1]

        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - self.cyl_radius**2
        disc = b * b - 4.0 * a * c

        solvable = (a > _EPS) & (disc >= 0.0)
        sq = np.sqrt(np.where(solvable, disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-b - sq) / (2.0 * a)
            t2 = (-b + sq) / (2.0 * a)

        # Smallest strictly-positive root (inf when neither root is positive).
        t1p = np.where(t1 > _EPS, t1, np.inf)
        t2p = np.where(t2 > _EPS, t2, np.inf)
        t = np.minimum(t1p, t2p)

        valid = solvable & np.isfinite(t) & (t < t_hit) & (t < ranges)
        return valid, t

    def _mark_ground(
        self,
        O: np.ndarray,
        D: np.ndarray,
        valid: np.ndarray,
        t: np.ndarray,
        grid: np.ndarray,
    ) -> None:
        """Rasterise valid ground hits into the polar (r, theta) occupancy grid.

        Hits falling inside the structural footprint (r < r_min) or beyond the
        maximum evaluation radius (r >= R_max) lie outside the annulus and are
        dropped; those outer rays are instead caught by the cylindrical wall.
        """
        if not np.any(valid):
            return
        tv = t[valid]
        x = O[valid, 0] + tv * D[valid, 0]
        y = O[valid, 1] + tv * D[valid, 1]

        r = np.hypot(x, y)
        theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)

        ir = np.floor((r - self.ground_r_min) / self.ground_r_res).astype(np.int64)
        ith = np.floor(theta / self.ground_dtheta).astype(np.int64)
        ith = np.clip(ith, 0, self.ground_n_az - 1)  # guard theta == 2*pi rounding
        in_bounds = (ir >= 0) & (ir < self.ground_n_r)
        grid[ir[in_bounds], ith[in_bounds]] = True

    def _mark_cylinder(
        self,
        O: np.ndarray,
        D: np.ndarray,
        valid: np.ndarray,
        t: np.ndarray,
        grid: np.ndarray,
    ) -> None:
        """Rasterise valid cylinder hits into the height/azimuth occupancy grid."""
        if not np.any(valid):
            return
        tv = t[valid]
        z = O[valid, 2] + tv * D[valid, 2]
        hx = O[valid, 0] + tv * D[valid, 0]
        hy = O[valid, 1] + tv * D[valid, 1]
        theta = np.mod(np.arctan2(hy, hx), 2.0 * np.pi)

        iz = np.floor((z - self.cyl_z_min) / self.cyl_z_res).astype(np.int64)
        ith = np.floor(theta / self.cyl_dtheta).astype(np.int64)
        ith = np.clip(ith, 0, self.cyl_n_az - 1)  # guard theta == 2*pi rounding
        in_bounds = (iz >= 0) & (iz < self.cyl_nz)
        grid[iz[in_bounds], ith[in_bounds]] = True

    # ----- grid cell -> world point helpers (for visualisation) -----
    def _ground_cells_to_points(self, grid: np.ndarray) -> np.ndarray:
        """World-space centres ``(N, 3)`` of occupied ground cells (z = 0).

        Cells are indexed ``(r_bin, theta_bin)``; centres are placed at the mid
        radius and mid azimuth of each occupied polar cell.
        """
        ir, ith = np.nonzero(grid)
        r = self.ground_r_min + (ir + 0.5) * self.ground_r_res
        theta = (ith + 0.5) * self.ground_dtheta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y, np.zeros_like(x)], axis=-1)

    def _cyl_cells_to_points(self, grid: np.ndarray) -> np.ndarray:
        """World-space centres ``(N, 3)`` of occupied cylinder-wall cells."""
        iz, ith = np.nonzero(grid)
        z = self.cyl_z_min + (iz + 0.5) * self.cyl_z_res
        theta = (ith + 0.5) * self.cyl_dtheta
        x = self.cyl_radius * np.cos(theta)
        y = self.cyl_radius * np.sin(theta)
        return np.stack([x, y, z], axis=-1)

    def _score(
        self, ground_grid: np.ndarray, cyl_grid: np.ndarray, individual: Individual
    ) -> float:
        """Normalised, weighted fitness from filled grids and sensor cost."""
        covered_cells = int(ground_grid.sum()) + int(cyl_grid.sum())
        m_cov = covered_cells / self._cells_total  # 0..1

        total_cost = sum(gene.sensor.price for gene in individual)
        c_norm = min(total_cost / self.max_budget, 1.0)  # 0..1

        return max(0.0, self.w_cov * m_cov - self.w_cost * c_norm)

    # ======================================================================
    # Public DEAP entry point
    # ======================================================================
    def evaluate_individual(self, individual: Individual) -> tuple[float]:
        """Return the DEAP fitness tuple for one sensor layout.

        ``fitness = w_cov * coverage_fraction - w_cost * cost_fraction``,
        clamped to be non-negative.
        """
        ground_grid = np.zeros((self.ground_n_r, self.ground_n_az), dtype=bool)
        cyl_grid = np.zeros((self.cyl_nz, self.cyl_n_az), dtype=bool)

        rays_chunks: list[np.ndarray] = []
        range_chunks: list[np.ndarray] = []

        for gene in individual:
            node_xyz = np.asarray(MOUNTING_GRAPH.nodes[gene.node_id]["pos"], dtype=float)
            local = self._generate_local_rays(gene.sensor)
            rays6 = self._transform_rays(
                local, node_xyz, gene.pitch, gene.roll, gene.yaw
            )
            rays_chunks.append(rays6)
            range_chunks.append(
                np.full(rays6.shape[0], gene.sensor.range_m, dtype=np.float32)
            )

        if not rays_chunks:
            self.last_ground_grid = ground_grid
            self.last_cyl_grid = cyl_grid
            return (0.0,)

        # --- One batched raycast for the entire genome (self-occlusion) ---
        rays6 = np.concatenate(rays_chunks, axis=0)
        ranges = np.concatenate(range_chunks, axis=0)
        t_hit = self._cast(rays6)

        O = rays6[:, 0:3].astype(np.float32, copy=False)
        D = rays6[:, 3:6].astype(np.float32, copy=False)

        g_valid, g_t = self._ground_intersection(O, D, t_hit, ranges)
        c_valid, c_t = self._cylinder_intersection(O, D, t_hit, ranges)
        self._mark_ground(O, D, g_valid, g_t, ground_grid)
        self._mark_cylinder(O, D, c_valid, c_t, cyl_grid)

        self.last_ground_grid = ground_grid
        self.last_cyl_grid = cyl_grid

        return (self._score(ground_grid, cyl_grid, individual),)

    # ======================================================================
    # Visualisation helper
    # ======================================================================
    # Ray strike categories (for colouring in a viewer).
    RAY_BLOCKED = 0  # chassis self-occlusion
    RAY_GROUND = 1  # reaches the ground plane
    RAY_CYLINDER = 2  # reaches the cylinder wall
    RAY_MISS = 3  # escapes without striking a target within range

    def coverage_debug(
        self,
        individual: Individual,
        max_rays_per_sensor: int = 250,
        include_misses: bool = False,
    ) -> dict:
        """Recompute coverage and return geometry for visualisation.

        Mirrors :meth:`evaluate_individual` exactly (same grids, same fitness)
        but additionally returns, for plotting:

        * ``ground_cover_points`` / ``cyl_cover_points`` -- world-space centres
          of the occupied S_gnd / S_cyl cells (i.e. exactly what fitness counts),
        * ``sensors`` -- per-gene subsampled ray segments (origin -> first
          strike) tagged with a ``RAY_*`` category for colouring.

        Args:
            max_rays_per_sensor: cap on rays drawn per sensor (evenly strided)
                so dense LiDARs stay legible.
            include_misses: keep rays that strike nothing within range.
        """
        ground_grid = np.zeros((self.ground_n_r, self.ground_n_az), dtype=bool)
        cyl_grid = np.zeros((self.cyl_nz, self.cyl_n_az), dtype=bool)

        empty = {
            "fitness": 0.0,
            "ground_grid": ground_grid,
            "cyl_grid": cyl_grid,
            "ground_cover_points": np.empty((0, 3)),
            "cyl_cover_points": np.empty((0, 3)),
            "sensors": [],
        }
        if not individual:
            return empty

        # --- Build rays, remembering each gene's slice into the batch ---
        rays_chunks: list[np.ndarray] = []
        range_chunks: list[np.ndarray] = []
        slices: list[tuple[int, int, object]] = []
        start = 0
        for gene in individual:
            node_xyz = np.asarray(MOUNTING_GRAPH.nodes[gene.node_id]["pos"], dtype=float)
            local = self._generate_local_rays(gene.sensor)
            rays6 = self._transform_rays(
                local, node_xyz, gene.pitch, gene.roll, gene.yaw
            )
            n = rays6.shape[0]
            rays_chunks.append(rays6)
            range_chunks.append(np.full(n, gene.sensor.range_m, dtype=np.float32))
            slices.append((start, start + n, gene))
            start += n

        rays6 = np.concatenate(rays_chunks, axis=0)
        ranges = np.concatenate(range_chunks, axis=0)
        t_hit = self._cast(rays6)
        O = rays6[:, 0:3].astype(np.float32, copy=False)
        D = rays6[:, 3:6].astype(np.float32, copy=False)

        # --- Intersections (drives both grids and ray categories) ---
        g_valid, g_t = self._ground_intersection(O, D, t_hit, ranges)
        c_valid, c_t = self._cylinder_intersection(O, D, t_hit, ranges)
        self._mark_ground(O, D, g_valid, g_t, ground_grid)
        self._mark_cylinder(O, D, c_valid, c_t, cyl_grid)
        self.last_ground_grid = ground_grid
        self.last_cyl_grid = cyl_grid

        # Per-ray candidate distances; +inf where that target is not reached.
        t_chassis = np.where(np.isfinite(t_hit) & (t_hit < ranges), t_hit, np.inf)
        t_ground = np.where(g_valid, g_t, np.inf)
        t_cyl = np.where(c_valid, c_t, np.inf)
        # Column order must match RAY_BLOCKED / RAY_GROUND / RAY_CYLINDER.
        t_stack = np.stack([t_chassis, t_ground, t_cyl], axis=1)

        sensors_dbg = []
        for s, e, gene in slices:
            n = e - s
            if n > max_rays_per_sensor:
                idx = np.linspace(s, e - 1, max_rays_per_sensor).astype(np.int64)
            else:
                idx = np.arange(s, e)

            rng = ranges[idx]
            sub = t_stack[idx]
            t_min = sub.min(axis=1)
            category = sub.argmin(axis=1).astype(np.int64)
            miss = ~np.isfinite(t_min)
            category[miss] = self.RAY_MISS

            # Draw misses out to the sensor range so direction stays visible.
            t_draw = np.where(miss, rng, t_min)
            origins = O[idx]
            endpoints = origins + t_draw[:, None] * D[idx]

            if not include_misses:
                keep = ~miss
                origins = origins[keep]
                endpoints = endpoints[keep]
                category = category[keep]

            sensors_dbg.append(
                {
                    "node_id": gene.node_id,
                    "sensor_type": gene.sensor.sensor_type,
                    "origin": O[s],
                    "ray_origins": origins,
                    "ray_endpoints": endpoints,
                    "ray_categories": category,
                }
            )

        return {
            "fitness": self._score(ground_grid, cyl_grid, individual),
            "ground_grid": ground_grid,
            "cyl_grid": cyl_grid,
            "ground_cover_points": self._ground_cells_to_points(ground_grid),
            "cyl_cover_points": self._cyl_cells_to_points(cyl_grid),
            "sensors": sensors_dbg,
        }


if __name__ == "__main__":
    # Smoke test: evaluate a one-sensor layout on the real mounting graph.
    from config.params import Gene, VALID_NODE_IDS
    from config.sensors import SENSOR_CATALOG

    evaluator = CoverageEvaluator()
    demo = [
        Gene(
            sensor=SENSOR_CATALOG[SensorType.LIDAR_16_CH],
            node_id=VALID_NODE_IDS[0],
            pitch=-10,
            roll=0,
            yaw=0,
        )
    ]
    score = evaluator.evaluate_individual(demo)
    g, cyl = evaluator.last_ground_grid, evaluator.last_cyl_grid
    print(f"fitness = {score[0]:.4f}")
    print(f"ground cells covered : {int(g.sum())} / {g.size}")
    print(f"cylinder cells covered: {int(cyl.sum())} / {cyl.size}")
