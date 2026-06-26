"""K-fold cross-validation of the BC(true)->PPO(true) pipeline (plan Phase E rigor).

Splits the 8-robot fleet into folds; each fold trains on the rest and evaluates the
held-out robots zero-shot, true-verified. Every robot is thus measured as unseen,
giving a fleet-wide generalization number with a CI across robots. True footprint
tables are cached on disk (fold-independent), so they are built once.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import shapes
from .bc import bc_pretrain
from .bc_true import cached_true_table, generate_true_demos
from .env import PlacementEnv
from .infer import evaluate_robot
from .policy import PlacementPolicy
from .reward import RewardModel
from .train_ppo import _true_fitness, train
from .verify_ceiling import optimum_layout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_OPT_CACHE: dict = {}


def optimum_fit(robot):
    """True-verified multi-start+LS optimum (the honest ceiling), cached per robot."""
    if robot not in _OPT_CACHE:
        layout, _ = optimum_layout(robot)
        _OPT_CACHE[robot] = _true_fitness(robot, layout)
    return _OPT_CACHE[robot]


def run_fold(held, all_robots, ppo_iters=50, seed=0):
    train_robots = [r for r in all_robots if r not in held]
    print(f"\n=== fold: held-out {held} ===")
    demos = generate_true_demos(train_robots, seed=seed)
    policy = PlacementPolicy().to(DEVICE)
    bc_pretrain(policy, demos, epochs=12, seed=seed)
    policy = train(
        val_robots=held,
        init_policy=policy,
        true_reward=True,
        iters=ppo_iters,
        episodes_per_iter=48,
        eval_every=ppo_iters,
        seed=seed,
    )

    env = PlacementEnv(RewardModel(device=DEVICE))
    out = {}
    for r in held:
        m = evaluate_robot(env, policy, r)  # greedy / best_of_n / verify-and-fallback
        out[r] = m
        fb = f" fallback={m['fallback']:.4f}" if m["fallback"] is not None else ""
        print(f"  held-out {r:18s} greedy={m['greedy']:.4f}  bestN={m['best_of_n']:.4f}"
              f"  final={m['final']:.4f}{fb}")
    return out


def _multiseed_summary(per_seed, opt, seeds):
    """Aggregate over (seeds x robots): the honest Phase-0 baseline vs the OPT ceiling."""
    robots = list(next(iter(per_seed.values())).keys())
    print(f"\n=== MULTI-SEED summary (seeds={seeds}) ===")
    print(f"  {'robot':18s} {'bestN(mu)':>10s} {'final(mu)':>10s} {'OPT':>8s} {'%OPT':>6s}")
    for r in robots:
        bn = np.mean([per_seed[s][r]["best_of_n"] for s in seeds])
        fn = np.mean([per_seed[s][r]["final"] for s in seeds])
        o = opt[r]
        pct = 100 * fn / o if o > 0 else float("nan")
        print(f"  {r:18s} {bn:10.4f} {fn:10.4f} {o:8.4f} {pct:6.0f}")
    print()
    for k in ["greedy", "best_of_n", "final"]:
        vals = np.array([per_seed[s][r][k] for s in seeds for r in robots])
        ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
        print(f"  fleet {k:10s} = {vals.mean():.4f} ± {ci:.4f} (n={len(vals)})")
    opt_mean = np.mean([opt[r] for r in robots])
    final_mean = np.mean([per_seed[s][r]["final"] for s in seeds for r in robots])
    print(f"  fleet OPT      = {opt_mean:.4f}   ->  policy(final) = {100*final_mean/opt_mean:.0f}% of OPT")


if __name__ == "__main__":
    import sys

    seeds = [int(s) for s in (sys.argv[1:] or ["0", "1", "2"])]
    robots = shapes.robot_names()
    for r in robots:  # pre-build/caches every robot's true table once
        cached_true_table(r)
    opt = {r: optimum_fit(r) for r in robots}  # OPT ceiling (seed-independent)
    print("OPT (multi-start+LS) per robot:", {r: round(v, 4) for r, v in opt.items()}, flush=True)
    folds = [robots[i : i + 2] for i in range(0, len(robots), 2)]  # 4 folds of 2

    per_seed = {}
    for seed in seeds:
        print(f"\n########## SEED {seed} ##########", flush=True)
        res = {}
        for held in folds:
            res.update(run_fold(held, robots, seed=seed))
        per_seed[seed] = res
    _multiseed_summary(per_seed, opt, seeds)
