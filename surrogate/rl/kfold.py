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
from .policy import PlacementPolicy
from .reward import RewardModel
from .train_ppo import _true_fitness, rollout, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        _, _, layout = rollout(env, policy, r, greedy=True)
        out[r] = _true_fitness(r, layout)
        print(f"  held-out {r:18s} zero-shot true_fit={out[r]:.4f}")
    return out


if __name__ == "__main__":
    robots = shapes.robot_names()
    # Pre-build/caches every robot's true table once.
    for r in robots:
        cached_true_table(r)
    folds = [robots[i : i + 2] for i in range(0, len(robots), 2)]  # 4 folds of 2

    results = {}
    for held in folds:
        results.update(run_fold(held, robots))

    vals = np.array(list(results.values()))
    ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
    print("\n=== k-fold zero-shot summary (every robot held out once) ===")
    for r, f in results.items():
        print(f"  {r:18s} {f:.4f}")
    print(
        f"\n  fleet mean zero-shot true_fit = {vals.mean():.4f} ± {ci:.4f} (95% CI, n={len(vals)})"
    )
