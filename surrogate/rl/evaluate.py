"""Phase E: zero-shot evaluation, true-verified, with inference-cost timing.

Compares the trained policy (one amortized rollout, no per-robot footprint table)
against greedy-on-surrogate and greedy-on-true (the ceiling) on held-out robots.
Every layout is scored by the real ``CoverageEvaluator``. The policy is reported as
the deterministic greedy(argmax) rollout plus a stochastic-rollout mean +/- 95% CI.
Timing isolates the amortization claim: encode-once + rollout vs building a footprint
table (surrogate or raycast) then greedy.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import torch

from .. import shapes
from .bc_true import build_true_table, greedy_over_masks
from .env import ORIENT_BINS, PlacementEnv
from .eval_baselines import spread_nodes
from .finetune import load_bc_policy
from .policy import PlacementPolicy
from .reward import SENSOR_BY_TYPE, RewardModel
from .train_ppo import _true_fitness, rollout

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _layout_from_sel(sel):
    from config.params import Gene

    return [
        Gene(
            sensor=SENSOR_BY_TYPE[t + 1],
            node_id=nid,
            pitch=ORIENT_BINS[o][0],
            roll=ORIENT_BINS[o][1],
            yaw=ORIENT_BINS[o][2],
        )
        for nid, t, o in sel
    ]


def eval_policy(policy, env, robot, n_samples=30, seed=0):
    """Greedy(argmax) rollout + stochastic-rollout mean/CI, true-verified + timed."""
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    _, _, layout = rollout(env, policy, robot, greedy=True)
    t_roll = time.perf_counter() - t0  # includes encode (env.reset -> set_robot)
    greedy_fit = _true_fitness(robot, layout)

    fits = []
    for _ in range(n_samples):
        _, _, lay = rollout(env, policy, robot, greedy=False)
        fits.append(_true_fitness(robot, lay))
    fits = np.array(fits)
    ci = 1.96 * fits.std(ddof=1) / np.sqrt(len(fits)) if len(fits) > 1 else 0.0
    return {
        "greedy": greedy_fit,
        "n": len(layout),
        "sample_mean": float(fits.mean()),
        "sample_ci": float(ci),
        "sample_best": float(fits.max()),
        "t_infer": t_roll,
    }


def eval_ceiling(robot, pool_size=100, seed=0, n_starts=40):
    """True-verified classical ceiling on one raycast table: plain greedy AND the
    honest baseline multi-start+LS (greedy is only a (1-1/e) lower bound)."""
    from .reward import SENSOR_BY_TYPE
    from .verify_ceiling import greedy, multistart_ls

    ev, g = shapes.build_evaluator(robot)
    pool = spread_nodes(g, pool_size, seed=seed)
    scorer = ev._scorer
    t0 = time.perf_counter()
    masks, cn, ct, co = build_true_table(ev, pool)  # raycast table (the cost)
    prices = np.array([SENSOR_BY_TYPE[int(t)].price for t in ct], dtype=float)
    g_rows, _ = greedy(masks, cn, prices, scorer)
    o_rows, _ = multistart_ls(masks, cn, prices, scorer, n_starts, seed)
    t = time.perf_counter() - t0

    def _verify(rows):
        lay = _layout_from_sel([(int(cn[i]), int(ct[i]) - 1, int(co[i])) for i in rows])
        return (ev.evaluate_individual(lay)[0] if lay else 0.0), len(lay)

    g_fit, _ = _verify(g_rows)
    o_fit, o_n = _verify(o_rows)
    return {"greedy": g_fit, "optimum": o_fit, "n_opt": o_n, "t_infer": t}


def eval_greedy_surrogate(robot, rm, pool_size=100, seed=0):
    from .bc import greedy_surrogate

    ev, g = shapes.build_evaluator(robot)
    pool = spread_nodes(g, pool_size, seed=seed)
    t0 = time.perf_counter()
    rm.set_robot(robot)  # GNN encode
    sel = greedy_surrogate(rm, pool)  # surrogate footprint table + greedy
    t = time.perf_counter() - t0
    layout = _layout_from_sel(sel)
    return {
        "fit": ev.evaluate_individual(layout)[0] if layout else 0.0,
        "n": len(layout),
        "t_infer": t,
    }


if __name__ == "__main__":
    import sys

    ckpt = DATA / (sys.argv[1] if len(sys.argv) > 1 else "rl_policy_ppo_true_bctrue.pt")
    held = sys.argv[2:] or ["rbwatcher", "rbvogui_xl"]
    print(f"policy: {ckpt.name}\nheld-out: {held}\n")

    policy = load_bc_policy(ckpt)
    rm = RewardModel(device=DEVICE)
    env = PlacementEnv(rm)

    hdr = (
        f"{'robot':16s} {'policy(bestN)':>13s} {'greedy_sur':>11s} "
        f"{'greedy_true':>12s} {'OPT(ms+LS)':>11s} | "
        f"{'t_pol':>7s} {'t_sur':>7s} {'t_opt':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in held:
        p = eval_policy(policy, env, r)
        gs = eval_greedy_surrogate(r, rm)
        c = eval_ceiling(r)  # greedy + honest multi-start+LS ceiling
        pol = p["sample_best"]  # best-of-N (true-verified)
        gap = pol / c["optimum"] if c["optimum"] > 0 else float("nan")
        print(
            f"{r:16s} {pol:13.4f} {gs['fit']:11.4f} "
            f"{c['greedy']:12.4f} {c['optimum']:11.4f} | "
            f"{p['t_infer']:6.2f}s {gs['t_infer']:6.2f}s {c['t_infer']:6.2f}s "
            f"  [policy {gap*100:.0f}% of OPT]"
        )
