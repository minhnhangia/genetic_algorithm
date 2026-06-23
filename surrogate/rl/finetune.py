"""PPO fine-tune driver from the BC-pretrained policy (plan Phase D).

Loads ``data/rl_policy_bc.pt`` into a ``PlacementPolicy`` and PPO-fine-tunes via
``train_ppo.train(init_policy=...)``. Overfit-one-robot gate first (true-verified
fitness should climb off the BC start), then the full fleet with hold-out.
"""

from __future__ import annotations

import pathlib
import sys

import torch

from .policy import PlacementPolicy
from .train_ppo import DEVICE, train

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
BC_CKPT = DATA / "rl_policy_bc.pt"


def load_bc_policy(ckpt: pathlib.Path = BC_CKPT) -> PlacementPolicy:
    policy = PlacementPolicy().to(DEVICE)
    policy.load_state_dict(torch.load(ckpt, map_location=DEVICE)["state_dict"])
    return policy


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "overfit"
    # `true` -> real raycast terminal reward; `bctrue` -> warm-start from the
    # greedy-on-true BC checkpoint (rl_policy_bc_true.pt) instead of the default.
    true_reward = "true" in sys.argv[2:]
    init_ckpt = DATA / (
        "rl_policy_bc_true.pt" if "bctrue" in sys.argv[2:] else "rl_policy_bc.pt"
    )
    args = [a for a in sys.argv[2:] if a not in ("true", "bctrue")]
    tag = f" [{'true' if true_reward else 'surrogate'}-reward, init={init_ckpt.stem}]"
    if mode == "overfit":
        robot = args[0] if args else "rbkairos"
        print(f"=== PPO-from-BC overfit gate ({robot}){tag} ===")
        train(
            single_robot=robot,
            init_policy=load_bc_policy(init_ckpt),
            iters=30,
            episodes_per_iter=32,
            eval_every=3,
            true_reward=true_reward,
        )
    else:  # full fleet, hold out the last two robots
        print(f"=== PPO-from-BC full-fleet fine-tune{tag} ===")
        policy = train(
            init_policy=load_bc_policy(init_ckpt),
            iters=60,
            episodes_per_iter=48,
            eval_every=5,
            true_reward=true_reward,
        )
        rtag = "true" if true_reward else "sur"
        itag = "_bctrue" if "bc_true" in init_ckpt.stem else ""
        out = f"rl_policy_ppo_{rtag}{itag}.pt"
        torch.save({"state_dict": policy.state_dict()}, DATA / out)
        print(f"\nsaved data/{out}")
