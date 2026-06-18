"""Single-sensor coverage footprints and wrap-safe orientation encoding.

Phase 0 of the footprint-surrogate plan. A *footprint* is one sensor's covered-cell
mask on the shared evaluation grid (ground 60x360 + cylinder 20x360 = 28,800 cells).
We get it by scoring a one-gene layout with the existing
:class:`~custom_toolbox.evaluate.evaluate_fitness_raycast.CoverageEvaluator`: the
inter-sensor body-occlusion path in ``_cast_with_bodies`` only triggers for >=2
sensors, so a single gene's grids ARE that sensor's footprint.

The surrogate will later predict these masks; here we only *extract* them (to build
labels and to gate the union approximation) and define the orientation features the
model/sampler will use.

Orientation encoding is by **direction vectors** (wrap-safe, no +/-180 yaw seam):

* A full-360 deg sensor is a surface of revolution about its local +z spin axis, so
  its coverage depends only on the spin-axis direction -- a 2-DOF unit vector. The
  third Euler DOF (taken here as roll) is redundant and is pinned to 0 when sampling.
* A directional sensor needs its bore (+x) plus an in-plane reference (+y) to pin the
  roll of the FOV footprint.
"""

from __future__ import annotations

import numpy as np

from config.params import Gene
from config.sensors import Sensor
from custom_toolbox.evaluate.evaluate_fitness_raycast import CoverageEvaluator
from custom_toolbox.evaluate.sensor_body import rotation_matrix

# A sensor is treated as omnidirectional (spin-axis only) at/above this FOV.
_FULL_CIRCLE_DEG = 360.0 - 1e-6


def is_omnidirectional(sensor: Sensor) -> bool:
    """True if the sensor's horizontal FOV spans the full circle (yaw redundant)."""
    return sensor.fov_horizontal_deg >= _FULL_CIRCLE_DEG


def sensor_footprint(
    evaluator: CoverageEvaluator,
    sensor: Sensor,
    node_id: int,
    pitch: float,
    roll: float,
    yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(ground_grid, cyl_grid)`` covered by ONE sensor at one pose.

    Scores a one-gene layout, so the returned grids carry no inter-sensor occlusion
    -- exactly the per-sensor footprint the surrogate models. Copies are returned
    because the evaluator overwrites ``last_*_grid`` on the next call.
    """
    gene = Gene(sensor=sensor, node_id=node_id, pitch=pitch, roll=roll, yaw=yaw)
    evaluator._compute_fitness([gene])  # sets last_ground_grid / last_cyl_grid
    return evaluator.last_ground_grid.copy(), evaluator.last_cyl_grid.copy()


def footprint_flat(ground_grid: np.ndarray, cyl_grid: np.ndarray) -> np.ndarray:
    """Flatten the two grids into one fixed-length ``(28800,)`` boolean mask."""
    return np.concatenate([ground_grid.reshape(-1), cyl_grid.reshape(-1)])


def orientation_features(
    sensor: Sensor, pitch: float, roll: float, yaw: float
) -> np.ndarray:
    """Wrap-safe orientation feature for the surrogate / sampler.

    Returns the spin axis (local +z in world) for omnidirectional sensors -- a
    complete 2-DOF descriptor -- and the bore (+x) concatenated with the in-plane
    +y reference for directional sensors (to pin roll). The leading 3 entries are
    always the spin axis so a single tensor layout works for both; directional
    sensors fill the trailing 3 with the +y reference, omnidirectional sensors
    leave them zero.
    """
    R = rotation_matrix(pitch, roll, yaw)
    spin_axis = R @ np.array([0.0, 0.0, 1.0])
    feats = np.zeros(6, dtype=np.float64)
    feats[:3] = spin_axis
    if not is_omnidirectional(sensor):
        feats[:3] = R @ np.array([1.0, 0.0, 0.0])  # bore for directional sensors
        feats[3:] = R @ np.array([0.0, 1.0, 0.0])  # +y reference pins roll
    return feats


def sample_orientation(
    sensor: Sensor, rng: np.random.Generator
) -> tuple[int, int, int]:
    """Sample a ``(pitch, roll, yaw)`` pose, dropping the redundant DOF.

    For omnidirectional sensors roll is pinned to 0 (coverage depends only on the
    spin axis spanned by pitch+yaw), so no samples are wasted on the redundant DOF.
    Directional sensors sample all three angles. Ranges mirror the GA's mechanical
    limits (pitch/roll +/-90, yaw +/-180).
    """
    pitch = int(rng.integers(-90, 91))
    yaw = int(rng.integers(-180, 181))
    roll = 0 if is_omnidirectional(sensor) else int(rng.integers(-90, 91))
    return pitch, roll, yaw
