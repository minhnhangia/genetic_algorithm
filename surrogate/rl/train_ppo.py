"""Minimal PPO for the constructive placement policy (plan Phase D).

Rollouts use the terminal dense surrogate reward; the policy reads the surrogate's
frozen node embeddings. Episodes are short (<=MAX_SENSORS), so returns are just the
discounted terminal reward and the advantage is ``return - value`` (no GAE needed).
Greedy (argmax) rollouts are scored by the **true** evaluator for honest reporting.

Gate first with ``train(single_robot=...)`` (overfit one robot -> the policy's
true-verified fitness should climb toward the greedy baseline), then train across
the fleet with hold-out robots.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from .. import shapes
from .env import PlacementEnv
from .policy import PlacementPolicy
from .reward import RewardModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EVAL_CACHE: dict = {}


def _true_fitness(robot: str, layout) -> float:
    if robot not in _EVAL_CACHE:
        _EVAL_CACHE[robot] = shapes.build_evaluator(robot)[0]
    if not layout:
        return 0.0
    return _EVAL_CACHE[robot].evaluate_individual(layout)[0]


def rollout(
    env: PlacementEnv, policy: PlacementPolicy, robot: str, greedy: bool = False
):
    obs = env.reset(robot)
    trans, done, term_r, layout = [], False, 0.0, []
    while not done:
        act, logp, val, raw = policy.act(obs, greedy=greedy)
        nxt, r, done, info = env.step(act)
        trans.append((obs, raw, logp, val))
        if done:
            term_r, layout = r, info.get("layout", [])
        obs = nxt
    return trans, term_r, layout


def ppo_update(policy, opt, batch, clip=0.2, epochs=4, c_v=0.5, c_ent=0.01):
    advs = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=DEVICE)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    rets = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
    logp_old = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=DEVICE)

    for _ in range(epochs):
        losses = []
        for i, (obs, raw, _ret, _adv, _lp) in enumerate(batch):
            logp, ent, val = policy.evaluate(obs, raw)
            ratio = torch.exp(logp - logp_old[i])
            s1, s2 = ratio * advs[i], torch.clamp(ratio, 1 - clip, 1 + clip) * advs[i]
            losses.append(-torch.min(s1, s2) + c_v * (val - rets[i]) ** 2 - c_ent * ent)
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


def train(
    single_robot: str | None = None,
    val_robots: list[str] | None = None,
    iters: int = 40,
    episodes_per_iter: int = 32,
    lr: float = 3e-4,
    gamma: float = 0.99,
    seed: int = 0,
    eval_every: int = 5,
    init_policy: PlacementPolicy | None = None,
    true_reward: bool = False,
):
    torch.manual_seed(seed)
    random.seed(seed)
    rm = RewardModel(device=DEVICE)
    env = PlacementEnv(rm, true_reward=true_reward)
    # Warm-start from a BC-pretrained policy when provided (PPO fine-tune).
    policy = init_policy if init_policy is not None else PlacementPolicy().to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    all_robots = shapes.robot_names()
    val_robots = val_robots or all_robots[-2:]
    if single_robot:
        train_robots, eval_robots = [single_robot], [single_robot]
    else:
        train_robots = [r for r in all_robots if r not in val_robots]
        eval_robots = val_robots
    print(f"train: {train_robots}\neval:  {eval_robots}\ndevice: {DEVICE}\n")

    for it in range(iters):
        batch, ep_rets = [], []
        for _ in range(episodes_per_iter):
            robot = random.choice(train_robots)
            trans, term_r, _ = rollout(env, policy, robot)
            T = len(trans)
            for k, (obs, raw, lp, val) in enumerate(trans):
                ret = (gamma ** (T - 1 - k)) * term_r
                batch.append((obs, raw, ret, ret - val, lp))
            ep_rets.append(term_r)
        loss = ppo_update(policy, opt, batch)

        if it % eval_every == 0 or it == iters - 1:
            evals = []
            for robot in eval_robots:
                _, _, layout = rollout(env, policy, robot, greedy=True)
                evals.append((robot, len(layout), _true_fitness(robot, layout)))
            ev = "  ".join(f"{r}:{f:.4f}(n{n})" for r, n, f in evals)
            print(
                f"iter {it:3d}  train_return={np.mean(ep_rets):+.4f}  "
                f"loss={loss:.3f}  | greedy-rollout true-fit  {ev}"
            )

    return policy


if __name__ == "__main__":
    # Overfit-one-robot gate: true-verified fitness should climb.
    print("=== overfit-one-robot gate (rbtheron) ===")
    train(single_robot="rbtheron", iters=40, episodes_per_iter=32)
