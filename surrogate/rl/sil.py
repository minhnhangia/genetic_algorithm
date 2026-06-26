"""Best-of-N internalization: POMO-style multi-rollout + best-of-N self-imitation.

Motivation (lit-grounded): best-of-N+verify is our biggest win but it is an *inference*
trick we throw away after scoring. The literature (ExIt 1705.08439, SIL 1806.05635,
RAFT 2304.06767, ReST 2308.08998, BOND 2407.14622, POMO 2010.16011) all say: feed the
search's best output back into the policy. Here, per training INSTANCE (a robot) we draw
K rollouts, true-verify each (terminal raycast), and:
  * POMO baseline: advantage_i = R_i - mean_k(R)  (shared baseline, no critic; low-variance
    REINFORCE -> attacks the binding constraint, which we diagnosed as variance).
  * SIL/RAFT term: pull the policy toward the BEST rollout's action sequence
    (-lambda * sum log pi(a* | s*)) -> internalize best-of-N into the single-sample policy.

Built on the COVERAGE-BLIND path (PlacementEnv/PlacementPolicy/train_ppo) so it is a clean
A/B vs the Phase-0 baseline (fleet best-of-N = 0.2474). New file only; nothing else changes.
Success = the GREEDY (single-rollout) number climbs toward best-of-N, and best-of-N quality is
reached at smaller N (cheaper inference) -- not "beat OPT" (that needs reversible actions).
"""

from __future__ import annotations

import numpy as np
import torch

from .. import shapes
from .bc import bc_pretrain
from .bc_true import cached_true_table, generate_true_demos
from .env import PlacementEnv
from .infer import evaluate_robot, policy_candidates
from .policy import PlacementPolicy
from .reward import RewardModel
from .train_ppo import DEVICE, _true_fitness, rollout


def sil_update(policy, opt, ppo_batch, sil_batch, lam=0.5, clip=0.2, epochs=4, c_ent=0.01):
    """Clipped-PG on shared-baseline advantages + a best-of-N self-imitation term.

    ppo_batch: list of (obs, raw, adv, logp_old).  sil_batch: list of (obs, raw) for the
    best rollouts (the imitation targets). No value loss -- POMO uses the shared baseline.
    """
    advs = torch.tensor([b[2] for b in ppo_batch], dtype=torch.float32, device=DEVICE)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    logp_old = torch.tensor([b[3] for b in ppo_batch], dtype=torch.float32, device=DEVICE)

    last = 0.0
    for _ in range(epochs):
        pg = []
        for i, (obs, raw, _adv, _lp) in enumerate(ppo_batch):
            logp, ent, _ = policy.evaluate(obs, raw)
            ratio = torch.exp(logp - logp_old[i])
            s1, s2 = ratio * advs[i], torch.clamp(ratio, 1 - clip, 1 + clip) * advs[i]
            pg.append(-torch.min(s1, s2) - c_ent * ent)
        pg_loss = torch.stack(pg).mean()
        if sil_batch:
            sil = torch.stack([-policy.evaluate(o, r)[0] for o, r in sil_batch]).mean()
        else:
            sil = torch.zeros((), device=DEVICE)
        loss = pg_loss + lam * sil
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    return last


def train_sil(val_robots, init_policy, iters=50, instances_per_iter=8, K=6,
              lr=3e-4, gamma=0.99, seed=0, eval_every=50, true_reward=True,
              single_robot=None, lam=0.5):
    """POMO + best-of-N self-imitation PPO. K rollouts per instance share a baseline;
    the best rollout is an imitation target. Rollout budget = instances_per_iter*K
    (default 48, matching the Phase-0 baseline's episodes_per_iter)."""
    import random

    torch.manual_seed(seed); random.seed(seed)
    env = PlacementEnv(RewardModel(device=DEVICE), true_reward=true_reward)
    policy = init_policy
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    all_robots = shapes.robot_names()
    if single_robot:
        train_robots, val_robots = [single_robot], [single_robot]
    else:
        train_robots = [r for r in all_robots if r not in val_robots]

    for it in range(iters):
        ppo_batch, sil_batch, ep_best = [], [], []
        for _ in range(instances_per_iter):
            robot = single_robot or random.choice(train_robots)
            rolls = [rollout(env, policy, robot) for _ in range(K)]
            Rs = [r[1] for r in rolls]
            base = float(np.mean(Rs))
            bi = int(np.argmax(Rs))
            for trans, term_r, _ in rolls:
                adv = term_r - base  # POMO shared baseline
                for obs, raw, lp, _val in trans:
                    ppo_batch.append((obs, raw, adv, lp))
            sil_batch.extend((obs, raw) for obs, raw, _lp, _val in rolls[bi][0])
            ep_best.append(max(Rs))
        loss = sil_update(policy, opt, ppo_batch, sil_batch, lam=lam)
        if it % eval_every == 0 or it == iters - 1:
            ev = []
            for r in val_robots:
                _, _, lay = rollout(env, policy, r, greedy=True)
                ev.append(f"{r}:{_true_fitness(r, lay):.4f}(n{len(lay)})")
            print(f"iter {it:3d} best_return={np.mean(ep_best):+.4f} loss={loss:.3f} "
                  f"| greedy {'  '.join(ev)}", flush=True)
    return policy


def sil_run_fold(held, all_robots, seed=0, iters=50, lam=0.5):
    train_robots = [r for r in all_robots if r not in held]
    print(f"\n=== SIL fold: held-out {held} (seed {seed}) ===", flush=True)
    demos = generate_true_demos(train_robots, seed=seed)
    pol = PlacementPolicy().to(DEVICE)
    bc_pretrain(pol, demos, epochs=12, seed=seed)
    pol = train_sil(val_robots=held, init_policy=pol, true_reward=True,
                    iters=iters, eval_every=iters, seed=seed, lam=lam)
    env = PlacementEnv(RewardModel(device=DEVICE))
    out = {}
    for r in held:
        m = evaluate_robot(env, pol, r)  # greedy / best_of_n(16) / fallback
        # N-sweep: best-of-4 measures whether quality is reached at cheaper inference
        c4 = policy_candidates(env, pol, r, n_samples=4)
        m["best_of_4"] = max(_true_fitness(r, lay) for lay in c4)
        out[r] = m
        print(f"  held-out {r:18s} greedy={m['greedy']:.4f}  bo4={m['best_of_4']:.4f}  "
              f"bestN={m['best_of_n']:.4f}  final={m['final']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    import sys

    from .kfold import _multiseed_summary, optimum_fit

    mode = sys.argv[1] if len(sys.argv) > 1 else "kfold"
    lam = float(__import__("os").environ.get("SIL_LAMBDA", "0.5"))
    if mode == "overfit":  # cheap gate: GREEDY true-fit must climb toward best-of-N
        robot = sys.argv[2] if len(sys.argv) > 2 else "rbkairos"
        print(f"=== SIL overfit gate ({robot}) lambda={lam} ===", flush=True)
        demos = generate_true_demos([robot])
        pol = PlacementPolicy().to(DEVICE)
        bc_pretrain(pol, demos, epochs=12)
        train_sil(val_robots=[robot], init_policy=pol, single_robot=robot,
                  true_reward=True, iters=30, eval_every=3, lam=lam)
    else:  # multi-seed k-fold vs OPT
        seeds = [int(s) for s in (sys.argv[2:] or ["0", "1", "2"])]
        robots = shapes.robot_names()
        for r in robots:
            cached_true_table(r)
        opt = {r: optimum_fit(r) for r in robots}
        folds = [robots[i : i + 2] for i in range(0, len(robots), 2)]
        per_seed = {}
        for seed in seeds:
            print(f"\n########## SIL SEED {seed} (lambda={lam}) ##########", flush=True)
            res = {}
            for held in folds:
                res.update(sil_run_fold(held, robots, seed=seed, lam=lam))
            per_seed[seed] = res
        _multiseed_summary(per_seed, opt, seeds)
