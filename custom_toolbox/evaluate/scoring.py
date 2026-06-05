"""Genome identity and the fitness blend.

Kept separate from the raycasting machinery because both are pure, cheap, and
independently testable: :func:`genome_key` defines when two layouts are
interchangeable (and thus cacheable), and :class:`FitnessScorer` turns a covered
cell count plus a layout's cost into the scalar the GA maximises.
"""

from __future__ import annotations

from config.params import Individual


def genome_key(individual: Individual) -> tuple:
    """Canonical, hashable signature of a layout for the fitness memo.

    Fitness depends only on the *multiset* of genes: coverage is a union over
    sensors (order-independent) and cost is a sum (order-independent). Sorting
    canonicalises gene order while keeping multiplicity, so two layouts differing
    only by gene order share a cache entry, but a repeated sensor still counts
    twice toward cost. ``sensor_type`` keys the sensor because the catalog maps
    type 1:1 to specs.
    """
    return tuple(
        sorted(
            (g.sensor.sensor_type.value, g.node_id, g.pitch, g.roll, g.yaw)
            for g in individual
        )
    )


class FitnessScorer:
    """Normalised, weighted blend of grid coverage and financial cost.

    ``fitness = w_cov * coverage_fraction - w_cost * cost_fraction``, clamped to
    be non-negative, where ``coverage_fraction = covered_cells / total_cells``
    and ``cost_fraction = min(total_price / max_budget, 1)``.
    """

    def __init__(
        self,
        w_cov: float,
        w_cost: float,
        max_budget: float,
        total_cells: int,
    ) -> None:
        self.w_cov = float(w_cov)
        self.w_cost = float(w_cost)
        self.max_budget = float(max_budget)
        self.total_cells = int(total_cells)

    def score(self, covered_cells: int, individual: Individual) -> float:
        m_cov = covered_cells / self.total_cells  # 0..1
        total_cost = sum(gene.sensor.price for gene in individual)
        c_norm = min(total_cost / self.max_budget, 1.0)  # 0..1
        return max(0.0, self.w_cov * m_cov - self.w_cost * c_norm)
