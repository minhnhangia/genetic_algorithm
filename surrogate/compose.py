"""Compose per-sensor footprints into a layout score.

Layout coverage is the union of per-sensor footprint masks (Phase-0 verified
near-exact: median 0%, p95 0.27% cell error), and the cost term is computed
*exactly* via the evaluator's
:class:`~custom_toolbox.evaluate.scoring.FitnessScorer`. A hard boolean union
drives greedy selection; a differentiable soft union is provided for later
gradient orientation-refinement / the surrogate.
"""

from __future__ import annotations

import numpy as np

from config.params import Individual
from custom_toolbox.evaluate.scoring import FitnessScorer


def union_mask(masks: list[np.ndarray]) -> np.ndarray | None:
    """Boolean OR of flat footprint masks (``None`` for an empty list)."""
    out: np.ndarray | None = None
    for m in masks:
        out = m.copy() if out is None else (out | m)
    return out


def covered_count(masks: list[np.ndarray]) -> int:
    """Number of cells covered by the union of ``masks``."""
    u = union_mask(masks)
    return 0 if u is None else int(u.sum())


def layout_fitness(
    masks: list[np.ndarray], layout: Individual, scorer: FitnessScorer
) -> float:
    """Exact GA fitness from the footprint union plus the exact cost term."""
    return scorer.score(covered_count(masks), layout)


def soft_expected_coverage(probs: list[np.ndarray]) -> float:
    """Differentiable expected covered-cell count from per-sensor cell probs.

    ``E[covered] = sum_c (1 - prod_i (1 - p_i(c)))`` -- the smooth analogue of the
    boolean union, for gradient-based orientation optimisation and the surrogate's
    training objective.
    """
    if not probs:
        return 0.0
    not_covered = np.ones_like(probs[0], dtype=np.float64)
    for p in probs:
        not_covered = not_covered * (1.0 - p)
    return float((1.0 - not_covered).sum())
