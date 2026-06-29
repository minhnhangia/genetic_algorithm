"""BC from a greedy-on-TRUE expert -- a stronger teacher to raise the ceiling.

Diagnosis (see overfit gates): BC anchors the policy to its teacher's quality and
PPO only improves locally, so cloning greedy-on-SURROGATE caps true fitness near
~0.24 while greedy-on-TRUE reaches ~0.33. Here we build a **true raycast footprint
table once per robot** (offline; inference stays a single amortized rollout), run
exact greedy over the policy's own action space (node x type x ORIENT_BINS) to make
expert demos, then reuse ``bc.bc_pretrain``. PPO fine-tune follows via ``finetune``.
"""

from __future__ import annotations

import numpy as np
import torch

from config.params import MAX_SENSORS_PER_INDIVIDUAL

from .. import shapes
from ..footprints import footprint_flat, sensor_footprint
from .bc import bc_pretrain
from .env import ORIENT_BINS
from .eval_baselines import spread_nodes
from .policy import PlacementPolicy
from .reward import SENSOR_BY_TYPE, RewardModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_true_table(evaluator, node_pool):
    """True flat footprints for every (node, type 1..3, ORIENT_BINS) pose.

    Returns ``(masks (M,n_cells) bool, cand_node, cand_type, cand_ori)``.
    """
    masks, cn, ct, co = [], [], [], []
    for nid in node_pool:
        for t in (1, 2, 3):
            sensor = SENSOR_BY_TYPE[t]
            for oi, (p, r, y) in enumerate(ORIENT_BINS):
                g, c = sensor_footprint(evaluator, sensor, int(nid), p, r, y)
                masks.append(footprint_flat(g, c))
                cn.append(int(nid))
                ct.append(t)
                co.append(oi)
    return (np.asarray(masks), np.asarray(cn), np.asarray(ct), np.asarray(co))


def greedy_over_masks(
    masks,
    cand_node,
    cand_type,
    cand_ori,
    scorer,
    sub_nodes,
    max_sensors=MAX_SENSORS_PER_INDIVIDUAL,
):
    """Exact marginal-fitness greedy over a precomputed mask table, restricted to
    ``sub_nodes``. Returns ordered ``(node_id, type_idx0, orient_bin)``."""
    sub = np.fromiter((cn in sub_nodes for cn in cand_node), bool, len(cand_node))
    prices = np.array([SENSOR_BY_TYPE[int(t)].price for t in cand_type])
    union = np.zeros(masks.shape[1], dtype=bool)
    used: set[int] = set()
    base_cost, cur_fit = 0.0, 0.0
    selected: list[tuple[int, int, int]] = []
    for _ in range(max_sensors):
        cov_frac = (masks | union).sum(axis=1) / scorer.total_cells
        cost_frac = np.minimum((base_cost + prices) / scorer.max_budget, 1.0)
        fit = np.maximum(0.0, scorer.w_cov * cov_frac - scorer.w_cost * cost_frac)
        avail = sub & np.fromiter(
            (cn not in used for cn in cand_node), bool, len(cand_node)
        )
        fit = np.where(avail, fit, -1.0)
        i = int(fit.argmax())
        if fit[i] <= cur_fit:
            break
        union = union | masks[i]
        used.add(int(cand_node[i]))
        base_cost += prices[i]
        cur_fit = fit[i]
        selected.append((int(cand_node[i]), int(cand_type[i]) - 1, int(cand_ori[i])))
    return selected


_TABLE_DIR = shapes.SHAPES_DIR.parent / "true_tables"


_TABLE_TAG = "ff"  # floor-fix: one-sided opaque floor + z>=0 candidate pool


def cached_true_table(robot, pool_size=100, seed=0):
    """Build (or load from disk) a robot's true footprint table; fold-independent.

    The filename encodes the orientation-grid size so changing ``ORIENT_BINS``
    invalidates stale tables (their ``co`` indices would point at the wrong pose).
    ``_TABLE_TAG`` further invalidates tables when the scoring/pool semantics change
    (here: the one-sided opaque floor + sub-floor node exclusion).
    """
    _TABLE_DIR.mkdir(exist_ok=True)
    path = _TABLE_DIR / f"{robot}_p{pool_size}_o{len(ORIENT_BINS)}_s{seed}_{_TABLE_TAG}.npz"
    if path.exists():
        z = np.load(path)
        n_cells = int(z["n_cells"])
        masks = np.unpackbits(z["masks"], axis=1)[:, :n_cells].astype(bool)
        return masks, z["cn"], z["ct"], z["co"]
    ev, g = shapes.build_evaluator(robot)
    pool = spread_nodes(g, pool_size, seed=seed)
    masks, cn, ct, co = build_true_table(ev, pool)
    np.savez_compressed(
        path,
        masks=np.packbits(masks, axis=1),
        cn=cn,
        ct=ct,
        co=co,
        n_cells=masks.shape[1],
    )
    return masks, cn, ct, co


def generate_true_demos(robots, pool_size=100, n_episodes=30, subset_size=40, seed=0):
    """Greedy-on-true demos: list of (robot, used_tuple, n_placed, target)."""
    rng = np.random.default_rng(seed)
    scorer = RewardModel(device=DEVICE).scorer  # same blend as training/eval
    demos = []
    for robot in robots:
        masks, cn, ct, co = cached_true_table(robot, pool_size, seed)
        pool = np.unique(cn)
        for _ in range(n_episodes):
            sub = set(
                int(x)
                for x in rng.choice(
                    pool, size=min(subset_size, len(pool)), replace=False
                )
            )
            selected = greedy_over_masks(masks, cn, ct, co, scorer, sub)
            used: list[int] = []
            for nid, tdx, oi in selected:
                demos.append((robot, tuple(used), len(used), ("place", nid, tdx, oi)))
                used.append(nid)
            demos.append((robot, tuple(used), len(used), ("stop",)))
        print(f"  {robot:18s} pool={len(pool)} table={masks.shape[0]} masks")
    return demos


if __name__ == "__main__":
    from .env import PlacementEnv
    from .train_ppo import _true_fitness, rollout

    all_robots = shapes.robot_names()
    val_robots = all_robots[-2:]
    train_robots = [r for r in all_robots if r not in val_robots]
    print(f"BC(true) train: {train_robots}\nheld-out: {val_robots}\n")

    demos = generate_true_demos(train_robots)
    print(f"\ngenerated {len(demos)} demonstration steps\n")
    policy = PlacementPolicy().to(DEVICE)
    bc_pretrain(policy, demos, epochs=12)

    env = PlacementEnv(RewardModel(device=DEVICE))
    print("\nBC(true) policy greedy-rollout true fitness:")
    for robot in train_robots[:2] + val_robots:
        _, _, layout = rollout(env, policy, robot, greedy=True)
        tag = "held-out" if robot in val_robots else "train"
        print(
            f"  {robot:18s} ({tag:8s}) n={len(layout)}  true_fit={_true_fitness(robot, layout):.4f}"
        )

    torch.save(
        {"state_dict": policy.state_dict()},
        shapes.SHAPES_DIR.parent / "rl_policy_bc_true.pt",
    )
    print("\nsaved data/rl_policy_bc_true.pt")
