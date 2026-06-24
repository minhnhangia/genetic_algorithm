"""Verify how tight the greedy-on-true 'ceiling' is vs a stronger search.

Greedy has only the (1-1/e) submodular guarantee, so it's a LOWER bound on the
optimum over the candidate table. Here we attack the same table (node pool x 3 types
x ORIENT_BINS, true raycast masks) with deterministic greedy, 2-swap local search,
and randomized multi-start greedy, and report whether anything beats greedy -- i.e.
how far below the true optimum our 'ceiling' might sit.
"""

from __future__ import annotations

import numpy as np

from .. import shapes
from .bc_true import cached_true_table
from .reward import SENSOR_BY_TYPE, RewardModel

MAX_SENSORS = 4


def _fit_of(masks, prices, rows, scorer):
    if not rows:
        return 0.0
    union = np.zeros(masks.shape[1], dtype=bool)
    for i in rows:
        union |= masks[i]
    cov = union.sum() / scorer.total_cells
    cost = min(sum(prices[i] for i in rows) / scorer.max_budget, 1.0)
    return max(0.0, scorer.w_cov * cov - scorer.w_cost * cost)


def best_addition(
    masks, cn, prices, scorer, base_union, base_cost, used_nodes, chunk=3000
):
    """Best single candidate to add to ``base`` (distinct node); returns (row, fit)."""
    n = masks.shape[0]
    valid = np.fromiter((c not in used_nodes for c in cn), bool, n)
    best_i, best_fit = -1, -1.0
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        cov = (masks[sl] | base_union).sum(1) / scorer.total_cells
        cost = np.minimum((base_cost + prices[sl]) / scorer.max_budget, 1.0)
        fit = np.where(
            valid[sl], np.maximum(0.0, scorer.w_cov * cov - scorer.w_cost * cost), -1.0
        )
        j = int(fit.argmax())
        if fit[j] > best_fit:
            best_fit, best_i = float(fit[j]), s + j
    return best_i, best_fit


def greedy(masks, cn, prices, scorer, first=None):
    rows, union, used, cost, cur = [], np.zeros(masks.shape[1], bool), set(), 0.0, 0.0
    for step in range(MAX_SENSORS):
        if step == 0 and first is not None:
            i, fit = first, _fit_of(masks, prices, [first], scorer)
        else:
            i, fit = best_addition(masks, cn, prices, scorer, union, cost, used)
        if fit <= cur and not (step == 0 and first is not None):
            break
        rows.append(i)
        union = union | masks[i]
        used.add(int(cn[i]))
        cost += prices[i]
        cur = fit
    return rows, _fit_of(masks, prices, rows, scorer)


def local_search(masks, cn, prices, scorer, rows):
    """2-swap: drop each selected sensor and optimally refill, until no gain."""
    rows = list(rows)
    cur = _fit_of(masks, prices, rows, scorer)
    improved = True
    while improved:
        improved = False
        for p in range(len(rows)):
            keep = rows[:p] + rows[p + 1 :]
            base = np.zeros(masks.shape[1], bool)
            for i in keep:
                base |= masks[i]
            bcost = sum(prices[i] for i in keep)
            used = {int(cn[i]) for i in keep}
            cand, _ = best_addition(masks, cn, prices, scorer, base, bcost, used)
            new = keep + [cand]
            f = _fit_of(masks, prices, new, scorer)
            if f > cur + 1e-9:
                rows, cur, improved = new, f, True
                break
    return rows, cur


def _scorer_for(masks):
    from custom_toolbox.evaluate.scoring import FitnessScorer

    return FitnessScorer(0.7, 0.3, 10000.0, total_cells=masks.shape[1])


def multistart_ls(masks, cn, prices, scorer, n_starts=40, seed=0):
    """Greedy + 2-swap local search, plus randomized restarts; returns (rows, fit).

    A stronger 'ceiling' than plain greedy (which is only a (1-1/e) lower bound and
    can stay trapped in a local optimum).
    """
    rng = np.random.default_rng(seed)
    rows, fit = local_search(
        masks, cn, prices, scorer, greedy(masks, cn, prices, scorer)[0]
    )
    for _ in range(n_starts):
        r, _ = greedy(
            masks, cn, prices, scorer, first=int(rng.integers(masks.shape[0]))
        )
        r, f = local_search(masks, cn, prices, scorer, r)
        if f > fit:
            rows, fit = r, f
    return rows, fit


def optimum_layout(robot, n_starts=40, seed=0):
    """Best layout found by multi-start+LS over the true table -> (genes, fitness).

    The honest ceiling baseline (replaces plain greedy-on-true).
    """
    from .evaluate import _layout_from_sel  # lazy: avoids import cycle

    masks, cn, ct, co = cached_true_table(robot)
    prices = np.array([SENSOR_BY_TYPE[int(t)].price for t in ct], dtype=float)
    rows, fit = multistart_ls(masks, cn, prices, _scorer_for(masks), n_starts, seed)
    sel = [(int(cn[i]), int(ct[i]) - 1, int(co[i])) for i in rows]
    return _layout_from_sel(sel), fit


def verify(robot, n_starts=40, seed=0):
    masks, cn, ct, co = cached_true_table(robot)
    prices = np.array([SENSOR_BY_TYPE[int(t)].price for t in ct], dtype=float)
    scorer = _scorer_for(masks)

    g_rows, g_fit = greedy(masks, cn, prices, scorer)
    ls_rows, ls_fit = local_search(masks, cn, prices, scorer, g_rows)
    _, best_fit = multistart_ls(masks, cn, prices, scorer, n_starts, seed)

    ub = g_fit / (1 - 1 / np.e)  # submodular upper bound on OPT
    return {
        "greedy": g_fit,
        "greedy+LS": ls_fit,
        "multistart+LS": best_fit,
        "ub_1-1/e": ub,
    }


if __name__ == "__main__":
    import sys

    robots = sys.argv[1:] or ["rbkairos", "rbsummit_xl", "rbwatcher", "rbtheron"]
    print(
        f"{'robot':16s} {'greedy':>8s} {'greedy+LS':>10s} {'multi+LS':>10s} "
        f"{'gain%':>7s} {'UB(1-1/e)':>10s}"
    )
    for r in robots:
        d = verify(r)
        gain = 100 * (d["multistart+LS"] - d["greedy"]) / max(d["greedy"], 1e-9)
        print(
            f"{r:16s} {d['greedy']:8.4f} {d['greedy+LS']:10.4f} "
            f"{d['multistart+LS']:10.4f} {gain:7.2f} {d['ub_1-1/e']:10.4f}"
        )
