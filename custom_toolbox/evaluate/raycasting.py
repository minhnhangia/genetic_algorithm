"""Ray sourcing for the coverage evaluator.

Two collaborators, split by responsibility:

* :class:`ChassisScene`   -- wraps the Open3D tensor raycaster around the robot
  chassis. It answers self-occlusion queries (:meth:`~ChassisScene.cast`) and
  reports the robot's structural footprint radius.
* :class:`SensorRayModel` -- sensor optics. It turns a layout's genes into
  world-frame ray bundles, caching the (FOV/resolution-derived) local ray
  template once per :class:`~config.sensors.SensorType`.

Coordinate convention (matches ``utils/visualization.py``):
    x = forward, y = left, z = up; positive pitch tilts the sensor UP.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from config.graph import MOUNTING_GRAPH
from config.params import Gene, Individual
from config.sensors import Sensor, SensorType

from .sensor_body import rotation_matrix, sensor_body_mesh


def cast_rays_against_meshes(
    rays6: np.ndarray, meshes: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Cast ``(M, 6)`` rays against an ad-hoc set of ``(verts, faces)`` meshes.

    Builds a throwaway raycasting scene from ``meshes`` and returns the ``(M,)``
    distance to the first hit (``+inf`` for a miss). Used for per-individual
    sensor-body occlusion, where the occluder set changes every layout and is
    far too small to be worth a persistent BVH.

    The meshes are merged and handed to Open3D as tensors directly; the legacy
    ``TriangleMesh`` round-trip is ~9 ms per call and dominates otherwise.
    """
    vertex_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    offset = 0
    for verts, faces in meshes:
        verts = np.asarray(verts, dtype=np.float32)
        vertex_blocks.append(verts)
        face_blocks.append(np.asarray(faces, dtype=np.uint32) + offset)
        offset += verts.shape[0]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.concatenate(vertex_blocks, axis=0)),
        o3d.core.Tensor(np.concatenate(face_blocks, axis=0)),
    )
    rays_t = o3d.core.Tensor(rays6, dtype=o3d.core.Dtype.Float32)
    return scene.cast_rays(rays_t)["t_hit"].numpy()


class ChassisScene:
    """Open3D raycasting scene for chassis self-occlusion."""

    def __init__(self, mesh_path: str) -> None:
        legacy_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(legacy_mesh.triangles) == 0:
            raise ValueError(f"Loaded chassis mesh has no triangles: {mesh_path}")
        legacy_mesh.compute_vertex_normals()

        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh))

        # Structural footprint: the chassis' maximum extent in the X/Y plane.
        # Used as the default inner radius of the ground disk (rays cannot reach
        # the ground beneath the robot).
        verts = np.asarray(legacy_mesh.vertices, dtype=float)
        self.footprint_radius = (
            float(np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2).max())
            if len(verts)
            else 0.0
        )

    def cast(self, rays6: np.ndarray) -> np.ndarray:
        """Cast ``(M, 6)`` ``[ox,oy,oz,dx,dy,dz]`` rays.

        Returns the ``(M,)`` distance to the first chassis hit (``+inf`` for rays
        that never strike it).
        """
        rays_t = o3d.core.Tensor(rays6, dtype=o3d.core.Dtype.Float32)
        return self._scene.cast_rays(rays_t)["t_hit"].numpy()


class SensorRayModel:
    """Sensor optics: cached local ray templates + world-frame assembly."""

    def __init__(self, mounting_graph=MOUNTING_GRAPH) -> None:
        self._graph = mounting_graph
        self._local_ray_cache: dict[SensorType, np.ndarray] = {}

    def local_rays(self, sensor: Sensor) -> np.ndarray:
        """Return cached unit direction vectors ``(N, 3)`` for a sensor type.

        Spherical -> Cartesian with x=forward, y=left, z=up. Computed once per
        ``SensorType`` and reused for every gene of that type.
        """
        cached = self._local_ray_cache.get(sensor.sensor_type)
        if cached is not None:
            return cached

        num_h = max(
            1, int(round(sensor.fov_horizontal_deg / sensor.horizontal_res_deg))
        )
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

    @staticmethod
    def _rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
        """Body->world rotation (see :func:`sensor_body.rotation_matrix`)."""
        return rotation_matrix(pitch, roll, yaw)

    def _transform(
        self,
        local_rays: np.ndarray,
        node_xyz: np.ndarray,
        pitch: float,
        roll: float,
        yaw: float,
    ) -> np.ndarray:
        """Rotate ``local_rays`` and attach the mounting origin.

        Returns an ``(N, 6)`` array laid out as ``[ox, oy, oz, dx, dy, dz]``
        ready for :meth:`ChassisScene.cast`.
        """
        R = self._rotation_matrix(pitch, roll, yaw).astype(np.float32)
        world_dirs = local_rays @ R.T  # (N, 3), float32

        n = world_dirs.shape[0]
        rays6 = np.empty((n, 6), dtype=np.float32)
        rays6[:, 0:3] = node_xyz.astype(np.float32)
        rays6[:, 3:6] = world_dirs
        return rays6

    def gene_bundles(
        self, individual: Individual
    ) -> list[tuple[Gene, np.ndarray, np.ndarray]]:
        """Per-gene ``(gene, rays6, ranges)`` ray bundles for a layout.

        ``rays6`` is ``(N, 6)`` world-frame rays and ``ranges`` is the ``(N,)``
        per-ray sensor range. Callers concatenate these for a single batched
        raycast (and may keep the per-gene split for visualisation).
        """
        bundles: list[tuple[Gene, np.ndarray, np.ndarray]] = []
        for gene in individual:
            node_xyz = np.asarray(self._graph.nodes[gene.node_id]["pos"], dtype=float)
            local = self.local_rays(gene.sensor)
            rays6 = self._transform(local, node_xyz, gene.pitch, gene.roll, gene.yaw)
            ranges = np.full(rays6.shape[0], gene.sensor.range_m, dtype=np.float32)
            bundles.append((gene, rays6, ranges))
        return bundles

    def body_mesh(self, gene: Gene) -> tuple[np.ndarray, np.ndarray]:
        """World-frame ``(vertices, faces)`` of ``gene``'s sensor body."""
        node_xyz = np.asarray(self._graph.nodes[gene.node_id]["pos"], dtype=float)
        return sensor_body_mesh(gene, node_xyz)
