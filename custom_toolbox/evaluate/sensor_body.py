"""Physical sensor-body geometry, shared by the evaluator and the visualiser.

A sensor's body is modelled as a short cylinder centred on its mounting point and
rotated by the gene's roll/pitch/yaw. The *same* mesh is used for two purposes,
so the picture always matches the physics:

* the ray-cast evaluator adds other sensors' bodies as occluders, and
* the 3D viewer draws each sensor as that cylinder.

Coordinate convention: x = forward, y = left, z = up; positive pitch tilts UP.
The body's axis is the sensor's local +Z (the spin axis of a vertical LiDAR).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from config.params import Gene


def rotation_matrix(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """Body->world rotation, repo convention (positive pitch = UP).

    Single source of truth for the orientation convention shared by the sensor
    ray bundle (:class:`~custom_toolbox.evaluate.raycasting.SensorRayModel`) and
    the sensor body. Equivalent to ``Rz(yaw) @ Ry(-pitch) @ Rx(roll)``; the
    negated pitch makes a forward ray map to ``z = +sin(pitch)``.
    """
    return Rotation.from_euler("ZYX", [yaw, -pitch, roll], degrees=True).as_matrix()


def _unit_cylinder(
    radius: float, height: float, sections: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """A closed cylinder centred at the origin with its axis along local +Z.

    Returns ``(vertices (V, 3), faces (F, 3))``. Includes top and bottom caps so
    that rays approaching along the axis are still occluded.
    """
    ang = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    x = radius * np.cos(ang)
    y = radius * np.sin(ang)
    hz = height / 2.0

    bottom = np.column_stack([x, y, np.full(sections, -hz)])
    top = np.column_stack([x, y, np.full(sections, hz)])
    centre_bottom = np.array([[0.0, 0.0, -hz]])
    centre_top = np.array([[0.0, 0.0, hz]])
    verts = np.vstack([bottom, top, centre_bottom, centre_top])

    s = sections
    i = np.arange(s)
    j = (i + 1) % s
    side_a = np.column_stack([i, j, s + j])
    side_b = np.column_stack([i, s + j, s + i])
    cap_bottom = np.column_stack([np.full(s, 2 * s), j, i])
    cap_top = np.column_stack([np.full(s, 2 * s + 1), s + i, s + j])
    faces = np.vstack([side_a, side_b, cap_bottom, cap_top]).astype(np.int64)

    return verts.astype(np.float64), faces


def sensor_body_mesh(
    gene: Gene, mount_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame ``(vertices, faces)`` of ``gene``'s sensor body.

    The local cylinder (sized by the sensor's ``body_radius_m`` /
    ``body_height_m``) is rotated by the gene's orientation and translated to its
    mounting point.
    """
    verts, faces = _unit_cylinder(
        gene.sensor.body_radius_m, gene.sensor.body_height_m
    )
    R = rotation_matrix(gene.pitch, gene.roll, gene.yaw)
    world_verts = verts @ R.T + np.asarray(mount_pos, dtype=float)
    return world_verts, faces
