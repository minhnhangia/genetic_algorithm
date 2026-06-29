"""Cost comparison: DRL policy (best-of-N, amortized) vs OPT (multi-start + local search).

Both methods consume the SAME cached true footprint table (a shared prerequisite, not timed).
We then time the SEARCH per robot, zero-shot:
  * policy  = encode (warm) + best-of-32 rollouts + 32 true-verify raycasts
  * OPT     = multi-start (40) + 2-swap local search over the table
Reports per-robot wall-clock + fitness, and renders cost (log) and cost-vs-quality figures.
Run:  python data/cost_comparison.py
"""
from __future__ import annotations

import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from surrogate import shapes
from surrogate.rl import explore
from surrogate.rl.bc_true import cached_true_table
from surrogate.rl.infer import policy_candidates
from surrogate.rl.train_ppo import _true_fitness
from surrogate.rl.verify_ceiling import optimum_layout

FOLDS = [
    ["rbkairos", "rbrobout"], ["rbsummit_xl", "rbsummit_steel"],
    ["rbtheron", "rbtheron_plus_top"], ["rbwatcher", "rbvogui_xl"],
]
N = 32
DATA = pathlib.Path(__file__).resolve().parent


def _ckpt_for(robot):
    for fold in FOLDS:
        if robot in fold:
            return explore._fold_ckpt(fold, 0)
    raise ValueError(robot)


def measure():
    robots = shapes.robot_names()
    rows = []
    print(f"{'robot':16s} {'t_policy':>9s} {'f_policy':>9s} {'t_OPT':>9s} {'f_OPT':>9s} {'speedup':>8s}")
    for r in robots:
        cached_true_table(r)  # shared prerequisite (not timed)
        pol, env, _ = explore.load_policy(_ckpt_for(r))
        policy_candidates(env, pol, r, n_samples=1)  # warm: encode + build cache

        t0 = time.perf_counter()
        cands = policy_candidates(env, pol, r, n_samples=N)
        f_pol = max(_true_fitness(r, lay) for lay in cands)
        t_pol = time.perf_counter() - t0

        t0 = time.perf_counter()
        _, f_opt = optimum_layout(r)
        t_opt = time.perf_counter() - t0

        rows.append((r, t_pol, f_pol, t_opt, f_opt))
        print(f"{r:16s} {t_pol:8.2f}s {f_pol:9.4f} {t_opt:8.2f}s {f_opt:9.4f} {t_opt / t_pol:7.1f}x",
              flush=True)
    return rows


def plot(rows):
    names = [r[0].replace("rb", "") for r in rows]
    t_pol = np.array([r[1] for r in rows])
    f_pol = np.array([r[2] for r in rows])
    t_opt = np.array([r[3] for r in rows])
    f_opt = np.array([r[4] for r in rows])
    x = np.arange(len(names)); w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

    # Panel 1: per-robot wall-clock (log)
    ax1.bar(x - w / 2, t_opt, w, label="OPT (multi-start + LS)", color="#2c3e50")
    ax1.bar(x + w / 2, t_pol, w, label=f"Policy (best-of-{N})", color="#27ae60")
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("wall-clock to produce a layout (s, log)")
    sp = np.mean(t_opt / t_pol)
    ax1.set_title(f"Inference cost per robot — policy is ~{sp:.0f}x cheaper than OPT")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3, which="both")

    # Panel 2: cost vs quality (fleet mean), excl rbrobout outlier
    keep = [i for i, r in enumerate(rows) if r[0] != "rbrobout"]
    cp, qp = t_pol[keep].mean(), 100 * (f_pol[keep] / f_opt[keep]).mean()
    co, qo = t_opt[keep].mean(), 100.0
    ax2.scatter([co], [qo], s=180, color="#2c3e50", label="OPT (ceiling)", zorder=3)
    ax2.scatter([cp], [qp], s=180, color="#27ae60", label=f"Policy best-of-{N}", zorder=3)
    ax2.annotate(f"OPT\n{co:.1f}s, 100%", (co, qo), textcoords="offset points", xytext=(-10, -28),
                 fontsize=9, ha="center")
    ax2.annotate(f"Policy\n{cp:.1f}s, {qp:.0f}% OPT", (cp, qp), textcoords="offset points",
                 xytext=(10, 10), fontsize=9, color="#1e8449")
    ax2.set_xlabel("wall-clock per robot (s)")
    ax2.set_ylabel("% of OPT quality (7 robots, excl rbrobout)")
    ax2.set_title("Cost vs quality — near-optimal at a fraction of the cost")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9, loc="lower right")
    ax2.set_ylim(min(qp - 8, 85), 103)

    fig.suptitle("Amortized DRL policy vs OPT: inference cost", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = DATA / "cost_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    plot(measure())
