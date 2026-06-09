"""Evaluation surfaces for the coverage fitness.

A surface is the geometry that sensor rays are scored against. Each one knows
how to, for a bundle of rays:

  1. :meth:`~EvaluationSurface.intersect` -- find the parametric hit distance and
     which rays actually reach the surface (before the chassis occludes them and
     within range),
  2. :meth:`~EvaluationSurface.mark` -- rasterise the surviving hits into its own
     boolean occupancy grid,
  3. :meth:`~EvaluationSurface.cells_to_points` -- map occupied cells back to
     world-space points (for visualisation).

The two concrete surfaces share a single maximum evaluation radius ``R_max``:

* :class:`GroundDisk`   -- an annular disk on ``Z = 0`` discretised in polar
  ``(r, theta)``, spanning ``r in [r_min, R_max]``.
* :class:`CylinderWall` -- a vertical wall standing at ``R_max`` (the rim of the
  ground disk) discretised in cylindrical ``(z, theta)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Epsilon rejecting grazing / zero-length parametric distances.
EPS = 1e-9


class EvaluationSurface(ABC):
    """A scoring surface with its own occupancy grid.

    Subclasses set :attr:`grid_shape`, :attr:`n_az` and :attr:`dtheta` and
    implement the three geometric primitives. All ray arrays are ``(N, 3)``
    world-frame origins ``O`` / directions ``D``; ``t_hit`` and ``ranges`` are
    ``(N,)`` per-ray chassis-occlusion and sensor-range cut-offs.
    """

    grid_shape: tuple[int, int]
    n_az: int
    dtheta: float

    def new_grid(self) -> np.ndarray:
        """A fresh, empty occupancy grid for this surface."""
        return np.zeros(self.grid_shape, dtype=bool)

    @property
    def n_cells(self) -> int:
        """Total number of grid cells (the denominator for coverage)."""
        return int(self.grid_shape[0] * self.grid_shape[1])

    @abstractmethod
    def intersect(
        self, O: np.ndarray, D: np.ndarray, t_hit: np.ndarray, ranges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(valid, t)``: the parametric hit distance per ray and a mask
        of the rays that strike this surface before occlusion and within range."""

    @abstractmethod
    def mark(
        self,
        O: np.ndarray,
        D: np.ndarray,
        valid: np.ndarray,
        t: np.ndarray,
        grid: np.ndarray,
    ) -> None:
        """Rasterise the valid hits into ``grid`` (mutated in place)."""

    @abstractmethod
    def cells_to_points(self, grid: np.ndarray) -> np.ndarray:
        """World-space centres ``(M, 3)`` of the occupied cells in ``grid``."""


class GroundDisk(EvaluationSurface):
    """Ground annulus on ``Z = 0``, polar ``(r, theta)`` grid.

    Hits inside the structural footprint (``r < r_min``) or beyond ``R_max`` lie
    outside the annulus and are dropped; those outer rays are caught by the
    cylindrical wall instead.
    """

    def __init__(
        self,
        max_radius: float,
        *,
        r_min: float = 0.0,
        r_res: float = 0.1,
        n_az: int = 360,
    ) -> None:
        self.max_radius = float(max_radius)
        # Guard against a footprint that would swallow the whole disk.
        self.r_min = min(float(r_min), self.max_radius - EPS)
        self.r_res = float(r_res)
        self.n_r = max(1, int(round((self.max_radius - self.r_min) / r_res)))
        self.n_az = int(n_az)
        self.dtheta = 2.0 * np.pi / self.n_az
        self.grid_shape = (self.n_r, self.n_az)

    def intersect(self, O, D, t_hit, ranges):
        dz = D[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -O[:, 2] / dz
        valid = (dz != 0.0) & (t > EPS) & (t < t_hit) & (t < ranges)
        return valid, t

    def mark(self, O, D, valid, t, grid):
        if not np.any(valid):
            return
        tv = t[valid]
        x = O[valid, 0] + tv * D[valid, 0]
        y = O[valid, 1] + tv * D[valid, 1]

        r = np.hypot(x, y)
        theta = np.mod(np.arctan2(y, x), 2.0 * np.pi)

        ir = np.floor((r - self.r_min) / self.r_res).astype(np.int64)
        ith = np.floor(theta / self.dtheta).astype(np.int64)
        ith = np.clip(ith, 0, self.n_az - 1)  # guard theta == 2*pi rounding
        in_bounds = (ir >= 0) & (ir < self.n_r)
        grid[ir[in_bounds], ith[in_bounds]] = True

    def cells_to_points(self, grid):
        ir, ith = np.nonzero(grid)
        r = self.r_min + (ir + 0.5) * self.r_res
        theta = (ith + 0.5) * self.dtheta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y, np.zeros_like(x)], axis=-1)


class CylinderWall(EvaluationSurface):
    """Vertical wall at radius ``R_max``, cylindrical ``(z, theta)`` grid."""

    def __init__(
        self,
        radius: float,
        *,
        z_min: float = 0.0,
        z_max: float = 4.0,
        z_res: float = 0.1,
        n_az: int = 360,
    ) -> None:
        self.radius = float(radius)
        self.z_min = float(z_min)
        self.z_res = float(z_res)
        self.nz = int(round((z_max - z_min) / z_res))
        self.n_az = int(n_az)
        self.dtheta = 2.0 * np.pi / self.n_az
        self.grid_shape = (self.nz, self.n_az)

    def intersect(self, O, D, t_hit, ranges):
        ox, oy = O[:, 0], O[:, 1]
        dx, dy = D[:, 0], D[:, 1]

        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - self.radius**2
        disc = b * b - 4.0 * a * c

        solvable = (a > EPS) & (disc >= 0.0)
        sq = np.sqrt(np.where(solvable, disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-b - sq) / (2.0 * a)
            t2 = (-b + sq) / (2.0 * a)

        # Smallest strictly-positive root (inf when neither root is positive).
        t1p = np.where(t1 > EPS, t1, np.inf)
        t2p = np.where(t2 > EPS, t2, np.inf)
        t = np.minimum(t1p, t2p)

        valid = solvable & np.isfinite(t) & (t < t_hit) & (t < ranges)
        return valid, t

    def mark(self, O, D, valid, t, grid):
        if not np.any(valid):
            return
        tv = t[valid]
        z = O[valid, 2] + tv * D[valid, 2]
        hx = O[valid, 0] + tv * D[valid, 0]
        hy = O[valid, 1] + tv * D[valid, 1]
        theta = np.mod(np.arctan2(hy, hx), 2.0 * np.pi)

        iz = np.floor((z - self.z_min) / self.z_res).astype(np.int64)
        ith = np.floor(theta / self.dtheta).astype(np.int64)
        ith = np.clip(ith, 0, self.n_az - 1)  # guard theta == 2*pi rounding
        in_bounds = (iz >= 0) & (iz < self.nz)
        grid[iz[in_bounds], ith[in_bounds]] = True

    def cells_to_points(self, grid):
        iz, ith = np.nonzero(grid)
        z = self.z_min + (iz + 0.5) * self.z_res
        theta = (ith + 0.5) * self.dtheta
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        return np.stack([x, y, z], axis=-1)
