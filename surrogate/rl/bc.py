"""Behavior-cloning warm-start from greedy-on-surrogate (plan Phase D rescue).

PPO-from-scratch can't explore the placement space (overfit gate stuck at 0). We
fix the start distribution by cloning a strong, cheap teacher -- **greedy over the
surrogate's footprints** -- whose placement sequences live in the policy's exact
action space (node, type, ORIENT_BINS). The policy is then PPO-fine-tuned. The
teacher is consistent with the surrogate reward, and ``greedy_surrogate`` doubles
as the Phase-4 cross-robot baseline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from config.params import MAX_SENSORS_PER_INDIVIDUAL, Gene

from .. import shapes
from .env import ORIENT_BINS, Obs
from .policy import PlacementPolicy
from .reward import SENSOR_BY_TYPE, RewardModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def greedy_surrogate(
    rm: RewardModel, node_subset, max_sensors: int = MAX_SENSORS_PER_INDIVIDUAL
):
    """Greedy layout over surrogate footprints; returns ordered (node_id, type_idx0, orient_bin).

    Vectorised marginal-fitness greedy over ``node_subset x 3 types x ORIENT_BINS``.
    Also the Phase-4 greedy-on-surrogate baseline.
    """
    cand_node, cand_type, cand_ori = [], [], []
    for n in node_subset:
        for t in (1, 2, 3):
            for oi in range(len(ORIENT_BINS)):
                cand_node.append(int(n))
                cand_type.append(t)
                cand_ori.append(oi)
    cand_node = np.array(cand_node)
    cand_type = np.array(cand_type)
    cand_ori = np.array(cand_ori)
    masks = rm.predict_masks_batch(
        cand_node, cand_type, [ORIENT_BINS[o] for o in cand_ori]
    )

    s = rm.scorer
    prices = np.array([SENSOR_BY_TYPE[int(t)].price for t in cand_type])
    union = np.zeros(rm.n_cells, dtype=bool)
    used: set[int] = set()
    base_cost, cur_fit = 0.0, 0.0
    selected: list[tuple[int, int, int]] = []

    for _ in range(max_sensors):
        newcov = (masks | union).sum(axis=1)
        cov_frac = newcov / s.total_cells
        cost_frac = np.minimum((base_cost + prices) / s.max_budget, 1.0)
        fit = np.maximum(0.0, s.w_cov * cov_frac - s.w_cost * cost_frac)
        fit = np.where([cn not in used for cn in cand_node], fit, -1.0)
        i = int(fit.argmax())
        if fit[i] <= cur_fit:
            break
        union = union | masks[i]
        used.add(int(cand_node[i]))
        base_cost += prices[i]
        cur_fit = fit[i]
        selected.append((int(cand_node[i]), int(cand_type[i]) - 1, int(cand_ori[i])))
    return selected


def generate_demos(robots, n_episodes: int = 30, subset_size: int = 60, seed: int = 0):
    """Greedy-on-surrogate demonstrations: list of (robot, used_tuple, n_placed, target)."""
    rm = RewardModel(device=DEVICE)
    rng = np.random.default_rng(seed)
    demos = []
    for robot in robots:
        rm.set_robot(robot)
        nodes = np.array(list(rm.graph.nodes()))
        for _ in range(n_episodes):
            subset = rng.choice(nodes, size=min(subset_size, len(nodes)), replace=False)
            selected = greedy_surrogate(rm, subset)
            used: list[int] = []
            for nid, tdx, oi in selected:
                demos.append((robot, tuple(used), len(used), ("place", nid, tdx, oi)))
                used.append(nid)
            demos.append((robot, tuple(used), len(used), ("stop",)))  # teach STOP
    return demos


def bc_pretrain(
    policy: PlacementPolicy, demos, epochs: int = 12, lr: float = 1e-3, seed: int = 0
):
    """Cross-entropy cloning of the teacher's (node|STOP, type, orient) decisions."""
    rm = RewardModel(device=DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    emb_cache: dict = {}

    def emb(robot):
        if robot not in emb_cache:
            rm.set_robot(robot)
            emb_cache[robot] = (rm._node_emb.detach(), rm.n_nodes)
        return emb_cache[robot]

    for ep in range(epochs):
        rng.shuffle(demos)
        total, nb = 0.0, 0
        for robot, used, n_placed, target in demos:
            node_emb, N = emb(robot)
            mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
            for u in used:
                mask[u] = True
            obs = Obs(node_emb=node_emb, used_mask=mask, n_placed=n_placed)
            H, ctx = policy._ctx(obs)
            place_logits = policy._place_logits(H, ctx, obs.used_mask).unsqueeze(0)
            if target[0] == "stop":
                loss = F.cross_entropy(place_logits, torch.tensor([N], device=DEVICE))
            else:
                _, nid, tdx, oi = target
                tl = policy.type_head(torch.cat([H[nid], ctx])).unsqueeze(0)
                te = policy.type_emb(torch.tensor(tdx, device=DEVICE))
                ol = policy.orient_head(torch.cat([H[nid], te, ctx])).unsqueeze(0)
                loss = (
                    F.cross_entropy(place_logits, torch.tensor([nid], device=DEVICE))
                    + F.cross_entropy(tl, torch.tensor([tdx], device=DEVICE))
                    + F.cross_entropy(ol, torch.tensor([oi], device=DEVICE))
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            nb += 1
        print(f"BC epoch {ep:2d}  loss={total / nb:.4f}")
    return policy


if __name__ == "__main__":
    from .env import PlacementEnv
    from .train_ppo import _true_fitness, rollout

    all_robots = shapes.robot_names()
    val_robots = all_robots[-2:]
    train_robots = [r for r in all_robots if r not in val_robots]
    print(f"BC train: {train_robots}\nheld-out: {val_robots}\n")

    demos = generate_demos(train_robots, n_episodes=30, subset_size=60)
    print(f"generated {len(demos)} demonstration steps\n")
    policy = PlacementPolicy().to(DEVICE)
    bc_pretrain(policy, demos, epochs=12)

    # Evaluate the BC policy's greedy rollout, true-verified.
    env = PlacementEnv(RewardModel(device=DEVICE))
    print("\nBC policy greedy-rollout true fitness:")
    for robot in train_robots[:2] + val_robots:
        _, _, layout = rollout(env, policy, robot, greedy=True)
        tag = "held-out" if robot in val_robots else "train"
        print(
            f"  {robot:18s} ({tag:8s}) n={len(layout)}  true_fit={_true_fitness(robot, layout):.4f}"
        )

    torch.save(
        {"state_dict": policy.state_dict()},
        shapes.SHAPES_DIR.parent / "rl_policy_bc.pt",
    )
    print("\nsaved data/rl_policy_bc.pt")
