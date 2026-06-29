"""Ceiling baselines for the RL policy, true-verified, on fleet robots.

For each robot: greedy-on-TRUE footprints (the no-ML upper bound) and
greedy-on-SURROGATE, both over a matched spatially-spread node subset, each scored
by the real ``CoverageEvaluator``. Printed beside the BC policy's rollout fitness so
the optimality gap is legible. select_spread_nodes uses a global graph, so we spread
over the robot's own node positions here.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import shapes
from ..select import build_candidates, greedy_select
from .bc import greedy_surrogate
from .reward import RewardModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def spread_nodes(graph, k: int, seed: int = 0, min_z: float = 0.0) -> list[int]:
    """Farthest-point sample ``k`` node ids by position (Euclidean spread).

    Nodes below ``min_z`` (the ground plane) are excluded: a sub-floor mount is
    physically invalid (chassis below the floor) and the evaluator's one-sided
    opaque floor scores it ~0, so it must not enter the candidate pool.
    """
    n = graph.number_of_nodes()
    pos = np.stack([graph.nodes[i]["pos"] for i in range(n)]).astype(np.float32)
    eligible = np.nonzero(pos[:, 2] >= min_z)[0]
    if len(eligible) == 0:  # degenerate fallback (no robot hits this)
        eligible = np.arange(n)
    epos = pos[eligible]
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(eligible)))]
    d = np.linalg.norm(epos - epos[chosen[0]], axis=1)
    while len(chosen) < min(k, len(eligible)):
        i = int(d.argmax())
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(epos - epos[i], axis=1))
    return [int(eligible[i]) for i in chosen]


def greedy_true_fit(evaluator, node_ids) -> float:
    cands = build_candidates(evaluator, node_ids)
    selected, _ = greedy_select(cands, scorer=evaluator._scorer)
    if not selected:
        return 0.0
    return evaluator.evaluate_individual([c.gene() for c in selected])[0]


def greedy_sur_fit(rm: RewardModel, evaluator, node_ids) -> float:
    from .env import ORIENT_BINS
    from .reward import SENSOR_BY_TYPE
    from config.params import Gene

    sel = greedy_surrogate(rm, node_ids)
    layout = [
        Gene(
            sensor=SENSOR_BY_TYPE[t + 1],
            node_id=nid,
            pitch=ORIENT_BINS[o][0],
            roll=ORIENT_BINS[o][1],
            yaw=ORIENT_BINS[o][2],
        )
        for nid, t, o in sel
    ]
    if not layout:
        return 0.0
    return evaluator.evaluate_individual(layout)[0]


if __name__ == "__main__":
    robots = ["rbkairos", "rbrobout", "rbwatcher", "rbvogui_xl"]
    held = {"rbwatcher", "rbvogui_xl"}
    k = 60
    rm = RewardModel(device=DEVICE)
    print(f"{'robot':18s} {'tag':9s} {'greedy_true':>12s} {'greedy_sur':>11s}")
    for r in robots:
        ev, g = shapes.build_evaluator(r)
        nodes = spread_nodes(g, k)
        rm.set_robot(r)
        gt = greedy_true_fit(ev, nodes)
        gs = greedy_sur_fit(rm, ev, nodes)
        tag = "held-out" if r in held else "train"
        print(f"{r:18s} {tag:9s} {gt:12.4f} {gs:11.4f}")
