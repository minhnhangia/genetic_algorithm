"""Ray-cast based fitness evaluation for the LiDAR layout GA.

This module replaces the analytic mock in ``evaluate_fitness_mock.py`` with a
physically grounded coverage estimate. For every sensor in an individual we:

  1. generate its local ray bundle from its FOV / resolution spec (cached),
  2. rotate + translate the bundle onto the sensor's mounting node,
  3. cast every ray against the robot chassis to find the self-occlusion
     distance ``t_hit``,
  4. analytically intersect each ray with the evaluation surfaces and keep a hit
     only if it occurs *before* the chassis blocks the ray and within range,
  5. rasterise the surviving hits into each surface's occupancy grid.

Fitness is a normalised, weighted blend of grid coverage and financial cost.

``CoverageEvaluator`` is a thin orchestrator that composes single-responsibility
collaborators (see the sibling modules):

  * :class:`~custom_toolbox.evaluate.raycasting.ChassisScene` -- self-occlusion
    raycasting + the robot's structural footprint radius.
  * :class:`~custom_toolbox.evaluate.raycasting.SensorRayModel` -- sensor optics:
    cached local ray templates and world-frame ray assembly.
  * :class:`~custom_toolbox.evaluate.surfaces.GroundDisk` /
    :class:`~custom_toolbox.evaluate.surfaces.CylinderWall` -- the two scoring
    surfaces (intersection + rasterisation + cell->point mapping). Both share a
    single maximum evaluation radius ``R_max``: the ground is a polar
    ``(r, theta)`` annulus out to ``R_max`` and the wall stands at ``R_max``.
  * :class:`~custom_toolbox.evaluate.scoring.FitnessScorer` /
    :func:`~custom_toolbox.evaluate.scoring.genome_key` -- the coverage/cost
    blend and the cache identity of a layout.

Coordinate convention: x = forward, y = left, z = up; positive pitch tilts UP.

Usage (notebook cell)::

    import custom_toolbox.evaluate.evaluate_fitness_raycast as evaluate
    evaluator = evaluate.CoverageEvaluator()
    toolbox.register("evaluate", evaluator.evaluate_individual)
"""

from __future__ import annotations

import numpy as np

from config.params import Individual

# Single source of truth for the chassis mesh location.
from generate_mounting_graph import CHASSIS_MESH_PATH

from .raycasting import ChassisScene, SensorRayModel, cast_rays_against_meshes
from .scoring import FitnessScorer, genome_key
from .surfaces import CylinderWall, GroundDisk


class CoverageEvaluator:
    """Stateful, reusable coverage evaluator for DEAP.

    The expensive setup -- loading the chassis mesh and building the raycasting
    scene -- happens once in ``__init__``. The per-individual cost is dominated
    by a single batched raycast plus vectorised intersection math, and identical
    genomes are memoised so they are scored only once.
    """

    # Ray strike categories (for colouring in a viewer; see :meth:`coverage_debug`).
    RAY_BLOCKED = 0  # chassis self-occlusion
    RAY_GROUND = 1  # reaches the ground plane
    RAY_CYLINDER = 2  # reaches the cylinder wall
    RAY_MISS = 3  # escapes without striking a target within range

    def __init__(
        self,
        mesh_path: str = str(CHASSIS_MESH_PATH),
        *,
        # --- Maximum evaluation radius R_max (shared by both surfaces) ---
        max_radius_m: float = 6.0,
        # --- Ground grid: annular disk in polar coords (r, theta) at Z = 0 ---
        ground_r_min_m: float | None = 0.0,  # inner radius; None => auto from mesh
        ground_r_res_m: float = 0.1,
        ground_azimuth_bins: int = 360,
        # --- Cylinder grid (height Z, azimuth theta at radius R_max) ---
        cyl_z_min_m: float = 0.0,
        cyl_z_max_m: float = 2.0,
        cyl_z_res_m: float = 0.1,
        cyl_azimuth_bins: int = 360,
        # --- Fitness blend ---
        w_cov: float = 0.7,
        w_cost: float = 0.3,
        max_budget: float = 10000.0,
    ) -> None:
        # Collaborators -------------------------------------------------------
        self._scene = ChassisScene(mesh_path)
        self._rays = SensorRayModel()

        self.max_radius = float(max_radius_m)
        r_min = (
            self._scene.footprint_radius
            if ground_r_min_m is None
            else float(ground_r_min_m)
        )
        self.ground = GroundDisk(
            self.max_radius,
            r_min=r_min,
            r_res=ground_r_res_m,
            n_az=ground_azimuth_bins,
        )
        self.cylinder = CylinderWall(
            self.max_radius,
            z_min=cyl_z_min_m,
            z_max=cyl_z_max_m,
            z_res=cyl_z_res_m,
            n_az=cyl_azimuth_bins,
        )
        self._scorer = FitnessScorer(
            w_cov,
            w_cost,
            max_budget,
            total_cells=self.ground.n_cells + self.cylinder.n_cells,
        )

        # Fitness memo: identical layouts (converged duplicates, carried-over
        # elites, no-op mutations) are scored once. ~35-40% of requested
        # evaluations are duplicates served from here in a typical run.
        self._fitness_cache: dict[tuple, tuple[float]] = {}

        # Most recent grids, kept for debugging / visualisation.
        self.last_ground_grid: np.ndarray | None = None
        self.last_cyl_grid: np.ndarray | None = None

    # ======================================================================
    # Core scoring pipeline
    # ======================================================================
    def _cast_with_bodies(
        self, bundles: list[tuple[object, np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cast every ray against the chassis *and* other sensors' bodies.

        Returns concatenated ``(rays6, ranges, t_hit)``. ``t_hit`` folds in two
        occluders: the static chassis (self-occlusion) and, per gene, the bodies
        of the *other* sensors in the layout. A sensor never occludes its own
        rays (they originate inside its own body), and a co-located sensor (same
        node) is skipped to avoid a degenerate zero-distance hit.
        """
        rays6 = np.concatenate([b[1] for b in bundles], axis=0)
        ranges = np.concatenate([b[2] for b in bundles], axis=0)
        t_hit = self._scene.cast(rays6)

        if len(bundles) >= 2:
            meshes = [self._rays.body_mesh(gene) for gene, _, _ in bundles]
            node_ids = [gene.node_id for gene, _, _ in bundles]
            offset = 0
            for i, (_, r6, _) in enumerate(bundles):
                n = r6.shape[0]
                others = [
                    meshes[j]
                    for j in range(len(meshes))
                    if j != i and node_ids[j] != node_ids[i]
                ]
                if others:
                    t_body = cast_rays_against_meshes(r6, others)
                    seg = slice(offset, offset + n)
                    np.minimum(t_hit[seg], t_body, out=t_hit[seg])
                offset += n

        return rays6, ranges, t_hit

    def _compute_fitness(self, individual: Individual) -> tuple[float]:
        """Raycast + score one layout (no cache). Updates ``last_*_grid``."""
        ground_grid = self.ground.new_grid()
        cyl_grid = self.cylinder.new_grid()

        bundles = self._rays.gene_bundles(individual)
        if not bundles:  # empty layout: no rays, no coverage
            self.last_ground_grid = ground_grid
            self.last_cyl_grid = cyl_grid
            return (0.0,)

        rays6, ranges, t_hit = self._cast_with_bodies(bundles)

        O = rays6[:, 0:3].astype(np.float32, copy=False)
        D = rays6[:, 3:6].astype(np.float32, copy=False)

        for surface, grid in ((self.ground, ground_grid), (self.cylinder, cyl_grid)):
            valid, t = surface.intersect(O, D, t_hit, ranges)
            surface.mark(O, D, valid, t, grid)

        self.last_ground_grid = ground_grid
        self.last_cyl_grid = cyl_grid

        covered = int(ground_grid.sum()) + int(cyl_grid.sum())
        return (self._scorer.score(covered, individual),)

    # ======================================================================
    # Public DEAP entry points
    # ======================================================================
    def evaluate_individual(self, individual: Individual) -> tuple[float]:
        """Return the DEAP fitness tuple for one sensor layout.

        Results are memoised by genome; a cache hit skips raycasting entirely
        (and leaves ``last_*_grid`` untouched).
        """
        key = genome_key(individual)
        cached = self._fitness_cache.get(key)
        if cached is not None:
            return cached
        result = self._compute_fitness(individual)
        self._fitness_cache[key] = result
        return result

    def evaluate_batch(self, individuals: list[Individual]) -> list[tuple[float]]:
        """Score a whole generation at once, deduplicating identical genomes.

        A drop-in for ``[evaluate_individual(i) for i in individuals]`` that only
        raycasts each *distinct, uncached* genome once and shares the result with
        every individual that maps to it -- so repeated layouts (cached across
        generations, plus the in-generation duplicates that selection and no-op
        mutations create) cost nothing. Bit-for-bit identical to the
        per-individual path; returns fitness tuples in input order.

        (A raycast that spans the whole generation was prototyped and measured
        ~20% *slower*: this coverage workload is memory-locality bound, so the
        small per-individual occupancy grids -- which stay cache-resident -- beat
        scattering hits into one stacked grid. Deduplication is the real win.)
        """
        results: list[tuple[float] | None] = [None] * len(individuals)

        # Cache lookup; group the misses by unique genome.
        pending: dict[tuple, list[int]] = {}
        representative: dict[tuple, Individual] = {}
        for i, ind in enumerate(individuals):
            key = genome_key(ind)
            cached = self._fitness_cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                pending.setdefault(key, []).append(i)
                representative.setdefault(key, ind)

        # Compute each distinct genome once, then fan the fitness back out.
        for key, idxs in pending.items():
            fit = self._compute_fitness(representative[key])
            self._fitness_cache[key] = fit
            for i in idxs:
                results[i] = fit

        return results  # type: ignore[return-value]

    def clear_cache(self) -> None:
        """Drop the memoised fitnesses (e.g. if scoring parameters change)."""
        self._fitness_cache.clear()

    # ======================================================================
    # Visualisation helper
    # ======================================================================
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
        ground_grid = self.ground.new_grid()
        cyl_grid = self.cylinder.new_grid()

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
        bundles = self._rays.gene_bundles(individual)
        slices: list[tuple[int, int, object]] = []
        start = 0
        for gene, rays6, ranges in bundles:
            n = rays6.shape[0]
            slices.append((start, start + n, gene))
            start += n

        # t_hit folds in both chassis and other-sensor-body occlusion, so rays
        # blocked by a neighbouring sensor are categorised RAY_BLOCKED too.
        rays6, ranges, t_hit = self._cast_with_bodies(bundles)
        O = rays6[:, 0:3].astype(np.float32, copy=False)
        D = rays6[:, 3:6].astype(np.float32, copy=False)

        # --- Intersections (drive both grids and ray categories) ---
        g_valid, g_t = self.ground.intersect(O, D, t_hit, ranges)
        c_valid, c_t = self.cylinder.intersect(O, D, t_hit, ranges)
        self.ground.mark(O, D, g_valid, g_t, ground_grid)
        self.cylinder.mark(O, D, c_valid, c_t, cyl_grid)
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

        covered = int(ground_grid.sum()) + int(cyl_grid.sum())
        return {
            "fitness": self._scorer.score(covered, individual),
            "ground_grid": ground_grid,
            "cyl_grid": cyl_grid,
            "ground_cover_points": self.ground.cells_to_points(ground_grid),
            "cyl_cover_points": self.cylinder.cells_to_points(cyl_grid),
            "sensors": sensors_dbg,
        }


if __name__ == "__main__":
    # Smoke test: evaluate a one-sensor layout on the real mounting graph.
    from config.params import Gene, VALID_NODE_IDS
    from config.sensors import SENSOR_CATALOG, SensorType

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
