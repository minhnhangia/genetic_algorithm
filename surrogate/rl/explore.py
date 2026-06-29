"""48h exploration: train+save a reusable true-gain policy, then inference-time levers
(Gumbel-Top-k decode, Neural-LNS destroy-repair). k-fold discards policies, so Phase 0 here
SAVES one true-gain coverage policy to disk as the substrate for the decode/LNS experiments.

Lit basis: NLNS destroy-repair (Hottung 2302.13797) reuses our constructive policy as the repair
operator; Gumbel-Top-k stochastic beam search (Kool 1903.06059) for diverse without-replacement decode.
New file; never overwrites existing checkpoints. See memory explore-48h-lns-decode-plan.
"""

from __future__ import annotations

import pathlib

import torch

from .. import shapes
from .coverage import (CoverageEnv, CoveragePolicy, coverage_bc_pretrain,
                       generate_coverage_demos, train_coverage)
from .infer import evaluate_robot
from .reward import RewardModel
from .train_ppo import DEVICE

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
HOLDOUT = ("rbwatcher", "rbvogui_xl")


def train_save(out="cov_truegain_h2.pt", holdout=HOLDOUT, seed=0, iters=50, true_gain=True):
    """Train a true-gain coverage policy (holdout = zero-shot eval) and SAVE it for reuse."""
    out_path = DATA / out
    assert not out_path.exists(), f"refusing to overwrite {out_path}"
    robots = shapes.robot_names()
    train_robots = [r for r in robots if r not in holdout]
    print(f"=== train_save true_gain={true_gain} holdout={holdout} -> {out} ===", flush=True)
    demos = generate_coverage_demos(train_robots, seed=seed)
    pol = CoveragePolicy().to(DEVICE)
    coverage_bc_pretrain(pol, demos, epochs=12, seed=seed, true_gain=true_gain)
    pol = train_coverage(val_robots=list(holdout), init_policy=pol, true_reward=True,
                         iters=iters, eval_every=iters, seed=seed, true_gain=true_gain)
    torch.save({"state_dict": pol.state_dict(), "holdout": list(holdout),
                "true_gain": true_gain, "seed": seed}, out_path)
    print(f"saved {out_path}", flush=True)

    env = CoverageEnv(RewardModel(device=DEVICE), true_gain=true_gain)
    for r in holdout:
        m = evaluate_robot(env, pol, r)
        print(f"  GATE held-out {r:14s} greedy={m['greedy']:.4f}  bestN={m['best_of_n']:.4f}  "
              f"final={m['final']:.4f}", flush=True)


def load_policy(ckpt="cov_truegain_h2.pt"):
    """Reload a saved coverage policy + its env (true_gain aware)."""
    blob = torch.load(DATA / ckpt, map_location=DEVICE)
    pol = CoveragePolicy().to(DEVICE)
    pol.load_state_dict(blob["state_dict"])
    env = CoverageEnv(RewardModel(device=DEVICE), true_gain=blob["true_gain"])
    return pol, env, blob


def decode_sweep(ckpt="cov_truegain_h2.pt", Ns=(8, 16, 32, 64, 128)):
    """Phase 1a: does i.i.d. best-of-N keep improving with N (variance headroom) or plateau
    (policy too peaked)? Draw max(N) samples ONCE, report best-of-first-n nested + greedy."""
    from .infer import policy_candidates
    from .train_ppo import _true_fitness

    pol, env, blob = load_policy(ckpt)
    print(f"=== decode N-sweep on {ckpt} (holdout {blob['holdout']}) ===", flush=True)
    for r in blob["holdout"]:
        cands = policy_candidates(env, pol, r, n_samples=max(Ns))  # [greedy, *sampled]
        fits = [_true_fitness(r, lay) for lay in cands]
        row = [f"greedy={fits[0]:.4f}"] + [f"bo{n}={max(fits[: n + 1]):.4f}" for n in Ns]
        print(f"  {r:14s} " + "  ".join(row), flush=True)


def _pool_index(env):
    return {int(n): i for i, n in enumerate(env.pool)}


def _repair(env, policy, robot, kept, greedy=False):
    """Roll out the policy CONDITIONED on a partial layout `kept` (list[Gene]). The kept
    sensors are replayed as forced actions, then the policy fills the rest. Returns layout."""
    from .env import Action, ORIENT_BINS

    obs = env.reset(robot)
    p2i = _pool_index(env)
    done = False
    for g in kept:
        if g.node_id not in p2i:
            continue
        oi = ORIENT_BINS.index((g.pitch, g.roll, g.yaw))
        a = Action(stop=False, node=p2i[g.node_id],
                   sensor_type=g.sensor.sensor_type.value, orient=oi)
        obs, _, done, _ = env.step(a)
        if done:
            break
    while not done:
        act, *_ = policy.act(obs, greedy=greedy)
        obs, _, done, _ = env.step(act)
    return list(env.rm.layout)


def lns(env, policy, robot, iters=30, k=1, T=0.04, seed=0):
    """Neural-LNS (NLNS): destroy k sensors, policy-repair, SA-accept, true-verify.
    Returns (best_layout, best_fit, n_raycasts)."""
    import math
    import random

    from .infer import policy_candidates
    from .train_ppo import _true_fitness

    rng = random.Random(seed)
    cands = policy_candidates(env, policy, robot, n_samples=8)  # init from best-of-8
    fits = [_true_fitness(robot, l) for l in cands]
    j = max(range(len(fits)), key=lambda i: fits[i])
    inc, inc_f = cands[j], fits[j]
    best, best_f = inc, inc_f
    rc = len(fits)
    for _ in range(iters):
        if len(inc) <= 1:
            kept = list(inc)
        else:
            drop = set(rng.sample(range(len(inc)), min(k, len(inc) - 1)))
            kept = [g for i, g in enumerate(inc) if i not in drop]
        rep = _repair(env, policy, robot, kept, greedy=False)
        f = _true_fitness(robot, rep); rc += 1
        if f >= inc_f or rng.random() < math.exp((f - inc_f) / T):
            inc, inc_f = rep, f
        if f > best_f:
            best, best_f = rep, f
    return best, best_f, rc


def lns_bench(ckpt="cov_truegain_h2.pt", iters=30, k=1):
    """Phase 2 gate: LNS vs i.i.d. best-of-N at MATCHED raycast budget, on the holdout."""
    from .infer import policy_candidates
    from .train_ppo import _true_fitness

    pol, env, blob = load_policy(ckpt)
    print(f"=== LNS vs best-of-N (matched budget) on {ckpt}  iters={iters} k={k} ===", flush=True)
    for r in blob["holdout"]:
        lay, f, rc = lns(env, pol, r, iters=iters, k=k)
        cands = policy_candidates(env, pol, r, n_samples=rc - 1)  # matched budget
        bo = max(_true_fitness(r, l) for l in cands)
        flag = "LNS WINS" if f > bo + 1e-4 else ("tie" if abs(f - bo) <= 1e-4 else "best-of-N wins")
        print(f"  {r:14s} LNS={f:.4f}  best-of-{rc}={bo:.4f}  (budget={rc} raycasts)  -> {flag}",
              flush=True)


def kfold_lns(iters=40, k=2, seed=0):
    """Phase 2 full validation: per fold, train+save a true-gain ckpt (zero-shot holdout), then
    LNS vs matched-budget best-of-N on the held-out pair. Aggregate fleet vs OPT."""
    import numpy as np

    from .infer import policy_candidates
    from .kfold import optimum_fit
    from .train_ppo import _true_fitness

    robots = shapes.robot_names()
    folds = [robots[i : i + 2] for i in range(0, len(robots), 2)]
    rows = {}
    for fold in folds:
        ckpt = ("cov_truegain_h2.pt" if list(fold) == ["rbwatcher", "rbvogui_xl"]
                else f"cov_truegain_{fold[0].replace('rb', '')}.pt")
        if not (DATA / ckpt).exists():
            train_save(out=ckpt, holdout=tuple(fold), seed=seed, true_gain=True)
        pol, env, _ = load_policy(ckpt)
        for r in fold:
            lay, f, rc = lns(env, pol, r, iters=iters, k=k)
            cands = policy_candidates(env, pol, r, n_samples=rc - 1)
            bo = max(_true_fitness(r, l) for l in cands)
            o = optimum_fit(r)
            rows[r] = (f, bo, o)
            print(f"  {r:16s} LNS={f:.4f}  bestN{rc}={bo:.4f}  OPT={o:.4f}  "
                  f"LNS%OPT={100 * f / o:.0f}", flush=True)
    lm = np.mean([v[0] for v in rows.values()])
    bm = np.mean([v[1] for v in rows.values()])
    om = np.mean([v[2] for v in rows.values()])
    print(f"\n  FLEET  LNS={lm:.4f} ({100 * lm / om:.0f}% OPT)  best-of-N={bm:.4f}  OPT={om:.4f}",
          flush=True)
    nz = [v for r, v in rows.items() if r != "rbrobout"]
    lmz = np.mean([v[0] for v in nz]); omz = np.mean([v[2] for v in nz])
    print(f"  FLEET(excl rbrobout outlier)  LNS={lmz:.4f} ({100 * lmz / omz:.0f}% OPT)", flush=True)


def dedup_analysis(ckpt="cov_truegain_h2.pt", N=64):
    """Phase 3a: how many of the N i.i.d. samples are DISTINCT layouts? A low distinct-fraction
    means best-of-N wastes raycasts on duplicates -> Gumbel-Top-k (without replacement) would reach
    the same quality at lower budget. High distinct-fraction -> little to gain, stop."""
    from .infer import policy_candidates
    from .train_ppo import _true_fitness

    pol, env, blob = load_policy(ckpt)
    print(f"=== dedup analysis on {ckpt}  N={N} ===", flush=True)
    for r in blob["holdout"]:
        cands = policy_candidates(env, pol, r, n_samples=N)
        sigs = [tuple(sorted((g.node_id, g.sensor.sensor_type.value, g.pitch, g.yaw)
                             for g in lay)) for lay in cands]
        uniq = len(set(sigs))
        fits = [_true_fitness(r, l) for l in cands]
        # best achievable using only the distinct layouts (= without-replacement upper bound proxy)
        best_sig = {}
        for s, f in zip(sigs, fits):
            best_sig[s] = max(best_sig.get(s, -1), f)
        print(f"  {r:14s} {uniq}/{N + 1} distinct ({100 * uniq / (N + 1):.0f}%)  "
              f"bestN={max(fits):.4f}  best-of-distinct={max(best_sig.values()):.4f}", flush=True)


def budget_frontier(Ns=(1, 4, 8, 16, 32, 64)):
    """Phase 3: cost/quality frontier. Reuse the 4 saved fold ckpts (zero-shot), draw 64 samples
    once per robot, report fleet %OPT at each best-of-N budget (incl + excl rbrobout outlier).
    Since samples are ~100% distinct, this is the real lever: quality vs raycast budget."""
    import numpy as np

    from .infer import policy_candidates
    from .kfold import optimum_fit
    from .train_ppo import _true_fitness

    robots = shapes.robot_names()
    folds = [robots[i : i + 2] for i in range(0, len(robots), 2)]
    per = {}
    for fold in folds:
        ckpt = ("cov_truegain_h2.pt" if list(fold) == ["rbwatcher", "rbvogui_xl"]
                else f"cov_truegain_{fold[0].replace('rb', '')}.pt")
        pol, env, _ = load_policy(ckpt)
        for r in fold:
            cands = policy_candidates(env, pol, r, n_samples=max(Ns))
            fits = [_true_fitness(r, l) for l in cands]  # fits[0]=greedy
            per[r] = {n: max(fits[:n]) for n in Ns}
    opt = {r: optimum_fit(r) for r in robots}
    print("=== budget frontier (fleet best-of-N %OPT vs N) ===", flush=True)
    for label, rset in [("all 8", robots),
                        ("excl robout", [r for r in robots if r != "rbrobout"])]:
        om = np.mean([opt[r] for r in rset])
        cells = [f"N{n}={np.mean([per[r][n] for r in rset]):.4f}"
                 f"({100 * np.mean([per[r][n] for r in rset]) / om:.0f}%)" for n in Ns]
        print(f"  {label:12s} " + "  ".join(cells), flush=True)


def _fold_ckpt(fold, seed):
    # "ff" = floor-fix: trained on the honest (one-sided floor + z>=0 pool) tables.
    tag = "h2" if list(fold) == ["rbwatcher", "rbvogui_xl"] else fold[0].replace("rb", "")
    base = "cov_truegain_ff_h2" if tag == "h2" else f"cov_truegain_ff_{tag}"
    return f"{base}.pt" if seed == 0 else f"{base}_s{seed}.pt"


def multiseed_frontier(seeds=(0, 1, 2), Ns=(1, 8, 16, 32, 64)):
    """Final rigor: multi-seed budget frontier with CIs. Seed 0 reuses existing ckpts;
    seeds 1,2 train fresh fold ckpts. Confirms the '~95% OPT at N=32 on 7/8' headline."""
    import numpy as np

    from .infer import policy_candidates
    from .kfold import optimum_fit
    from .train_ppo import _true_fitness

    robots = shapes.robot_names()
    folds = [robots[i : i + 2] for i in range(0, len(robots), 2)]
    opt = {r: optimum_fit(r) for r in robots}
    per_seed = {}
    for seed in seeds:
        per = {}
        for fold in folds:
            ckpt = _fold_ckpt(fold, seed)
            if not (DATA / ckpt).exists():
                train_save(out=ckpt, holdout=tuple(fold), seed=seed, true_gain=True)
            pol, env, _ = load_policy(ckpt)
            for r in fold:
                cands = policy_candidates(env, pol, r, n_samples=max(Ns))
                fits = [_true_fitness(r, l) for l in cands]
                per[r] = {n: max(fits[:n]) for n in Ns}
        per_seed[seed] = per
    rset = [r for r in robots if r != "rbrobout"]
    om = np.mean([opt[r] for r in rset])
    print(f"\n=== MULTISEED budget frontier (excl rbrobout; mean%OPT +/- CI over seeds x robots) "
          f"seeds={list(seeds)} ===", flush=True)
    for n in Ns:
        vals = np.array([per_seed[s][r][n] for s in seeds for r in rset])
        m, ci = vals.mean(), 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
        print(f"  N{n:>3} = {m:.4f} +/- {ci:.4f}  ({100 * m / om:.0f}% OPT)  n={len(vals)}", flush=True)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "train_save"
    if mode == "train_save":
        train_save()
    elif mode == "decode_sweep":
        decode_sweep()
    elif mode == "lns_bench":
        iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        lns_bench(iters=iters, k=k)
    elif mode == "kfold_lns":
        kfold_lns()
    elif mode == "dedup":
        dedup_analysis()
    elif mode == "budget_frontier":
        budget_frontier()
    elif mode == "multiseed_frontier":
        multiseed_frontier()
    else:
        raise SystemExit(f"unknown mode {mode}")
