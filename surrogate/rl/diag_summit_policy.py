"""Confirm the rbsummit_xl failure mechanism: retrain its fold (held out) and dump
the zero-shot policy's actual layout -- coverage, cost, and break-even -- to show it
places full-count/full-cost sensors whose coverage falls below the fitness floor.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import shapes
from .bc import bc_pretrain
from .bc_true import generate_true_demos
from .env import ORIENT_BINS, PlacementEnv
from .policy import PlacementPolicy
from .reward import SENSOR_BY_TYPE, RewardModel
from .train_ppo import rollout, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HELD = ["rbsummit_xl", "rbsummit_steel"]


def layout_breakdown(robot, layout):
    ev, _ = shapes.build_evaluator(robot)
    scorer = ev._scorer
    if layout:
        d = ev.coverage_debug(layout)
        cov = (int(d["ground_grid"].sum()) + int(d["cyl_grid"].sum())) / scorer.total_cells
    else:
        cov = 0.0
    fit = ev.evaluate_individual(layout)[0] if layout else 0.0
    cost = float(sum(g.sensor.price for g in layout))
    cost_frac = min(cost / scorer.max_budget, 1.0)
    breakeven_cov = scorer.w_cost * cost_frac / scorer.w_cov
    return fit, cost, cost_frac, breakeven_cov, cov


if __name__ == "__main__":
    train_robots = [r for r in shapes.robot_names() if r not in HELD]
    print(f"retrain fold, held-out {HELD}\n")
    demos = generate_true_demos(train_robots)
    policy = PlacementPolicy().to(DEVICE)
    bc_pretrain(policy, demos, epochs=12)
    policy = train(val_robots=HELD, init_policy=policy, true_reward=True,
                   iters=50, episodes_per_iter=48, eval_every=50)

    env = PlacementEnv(RewardModel(device=DEVICE))
    for robot in HELD:
        _, _, layout = rollout(env, policy, robot, greedy=True)
        fit, cost, cost_frac, be, cov = layout_breakdown(robot, layout)
        print(f"\n=== {robot} zero-shot policy layout ===")
        print(f"  n={len(layout)}  true_fit={fit:.4f}  cost={cost:.0f} (cost_frac={cost_frac:.3f})")
        print(f"  true coverage={cov}  break-even coverage needed={be:.3f}")
        for g in layout:
            print(f"    node={g.node_id:5d} type={g.sensor.sensor_type.value} "
                  f"price={g.sensor.price:.0f} pitch={g.pitch} yaw={g.yaw}")
