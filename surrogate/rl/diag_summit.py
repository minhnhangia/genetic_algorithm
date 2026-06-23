"""Diagnose why rbsummit_xl fails zero-shot (places n=3 but true_fit=0).

Contrasts rbsummit_xl with its fold-mate rbsummit_steel (trained identically, but
generalizes) and a healthy reference. For each robot reports: greedy-true ceiling
(coverage vs cost), greedy-on-surrogate (true-verified + what the surrogate THOUGHT
it scored), and the surrogate's footprint fidelity on that robot (mean IoU + coverage
bias). High ceiling + low surrogate fidelity => perception failure; low ceiling =>
just a hard robot; surrogate over-predicts coverage => policy chases phantom coverage.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import shapes
from .bc import greedy_surrogate
from .bc_true import cached_true_table, greedy_over_masks
from .env import ORIENT_BINS
from .evaluate import _layout_from_sel
from .reward import SENSOR_BY_TYPE, RewardModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cov_cost(masks_sel, prices_sel, scorer):
    union = np.zeros(masks_sel.shape[1], dtype=bool) if len(masks_sel) else np.zeros(scorer.total_cells, bool)
    for m in masks_sel:
        union |= m
    cov_frac = union.sum() / scorer.total_cells
    cost = float(sum(prices_sel))
    return cov_frac, cost, scorer.w_cov * cov_frac - scorer.w_cost * min(cost / scorer.max_budget, 1.0)


def diagnose(robot, rm):
    ev, g = shapes.build_evaluator(robot)
    scorer = ev._scorer
    masks, cn, ct, co = cached_true_table(robot)
    pool = set(int(x) for x in np.unique(cn))

    # --- greedy-true ceiling ---
    sel_t = greedy_over_masks(masks, cn, ct, co, scorer, pool)
    idx = {(int(cn[i]), int(ct[i]) - 1, int(co[i])): i for i in range(len(cn))}
    rows = [idx[s] for s in sel_t]
    cov_t, cost_t, fit_t = _cov_cost(masks[rows], [SENSOR_BY_TYPE[ct[r]].price for r in rows], scorer)
    true_fit_t = ev.evaluate_individual(_layout_from_sel(sel_t))[0] if sel_t else 0.0

    # --- greedy-on-surrogate (true-verified) + surrogate's own belief ---
    rm.set_robot(robot)
    sel_s = greedy_surrogate(rm, list(pool))
    true_fit_s = ev.evaluate_individual(_layout_from_sel(sel_s))[0] if sel_s else 0.0

    # --- surrogate fidelity on this robot: IoU + coverage bias over the table ---
    oris = [ORIENT_BINS[int(o)] for o in co]
    pred = rm.predict_masks_batch(cn, ct, oris)  # (M, n_cells) bool
    inter = (pred & masks).sum(1)
    union = (pred | masks).sum(1)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0).mean()
    pred_cov = pred.sum(1).mean() / scorer.total_cells
    true_cov = masks.sum(1).mean() / scorer.total_cells

    return {
        "ceiling_fit": true_fit_t, "ceiling_cov": cov_t, "ceiling_cost": cost_t, "ceiling_n": len(sel_t),
        "gsur_truefit": true_fit_s, "gsur_n": len(sel_s),
        "iou": float(iou), "pred_cov": float(pred_cov), "true_cov": float(true_cov),
    }


if __name__ == "__main__":
    rm = RewardModel(device=DEVICE)
    robots = ["rbsummit_xl", "rbsummit_steel", "rbkairos"]
    print(f"{'robot':16s} {'ceil_fit':>9s} {'ceil_cov':>9s} {'ceil_cost':>10s} "
          f"{'gsur_fit':>9s} | {'surr_IoU':>9s} {'pred_cov':>9s} {'true_cov':>9s}")
    print("-" * 96)
    for r in robots:
        d = diagnose(r, rm)
        print(f"{r:16s} {d['ceiling_fit']:9.4f} {d['ceiling_cov']:9.3f} {d['ceiling_cost']:10.0f} "
              f"{d['gsur_truefit']:9.4f} | {d['iou']:9.3f} {d['pred_cov']:9.3f} {d['true_cov']:9.3f}")
