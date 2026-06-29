"""Render the DRL policy's best-of-N layout for each robot via
``utils.visualization.visualize_best_layout``.

Each robot is rendered ZERO-SHOT: it uses the fold checkpoint in which that robot was
held out during the 48h exploration (data/cov_truegain_*.pt), so the layout is produced
by a policy that never trained on it.

Usage
-----
Headless (exports a .glb per robot to data/policy_layouts/):
    python visualize_policy_layouts.py                # all robots
    python visualize_policy_layouts.py rbkairos rbwatcher

In a notebook (inline 3D viewer):
    from visualize_policy_layouts import show_policy_layout
    show_policy_layout("rbkairos")                    # or show_rays=False, show_arrows=True, ...
"""

from __future__ import annotations

import pathlib

import trimesh

from surrogate import shapes
from surrogate.rl import explore
from surrogate.rl.infer import policy_candidates
from surrogate.rl.train_ppo import _true_fitness
from utils.visualization import visualize_best_layout

OUT = pathlib.Path(__file__).resolve().parent / "data" / "policy_layouts"
FOLDS = [
    ["rbkairos", "rbrobout"],
    ["rbsummit_xl", "rbsummit_steel"],
    ["rbtheron", "rbtheron_plus_top"],
    ["rbwatcher", "rbvogui_xl"],
]


def _ckpt_for(robot: str) -> str:
    """The fold checkpoint in which `robot` was held out (zero-shot)."""
    for fold in FOLDS:
        if robot in fold:
            return explore._fold_ckpt(fold, 0)
    raise ValueError(f"unknown robot {robot}")


def policy_layout(robot: str, n_samples: int = 32):
    """Best-of-N policy layout for `robot`, true-verified. Returns (Individual, fitness)."""
    pol, env, _ = explore.load_policy(_ckpt_for(robot))
    cands = policy_candidates(env, pol, robot, n_samples=n_samples)
    fits = [_true_fitness(robot, lay) for lay in cands]
    j = max(range(len(fits)), key=lambda i: fits[i])
    return cands[j], fits[j]


def _robot_assets(robot: str):
    evaluator, graph = shapes.build_evaluator(robot)
    mesh = trimesh.load(shapes.load_manifest()[robot]["mesh_path"], force="mesh")
    return evaluator, graph, mesh


def show_policy_layout(robot: str, n_samples: int = 32, **layout_kwargs):
    """Inline 3D viewer (notebook) of the policy's layout for one robot."""
    layout, _ = policy_layout(robot, n_samples)
    ev, graph, mesh = _robot_assets(robot)
    return visualize_best_layout(
        layout, evaluator=ev, mounting_graph=graph, mesh=mesh, **layout_kwargs
    )


def render_robot(robot: str, n_samples: int = 32, save: bool = True):
    """Build the policy layout scene for `robot`; export a .glb when `save`."""
    layout, fit = policy_layout(robot, n_samples)
    ev, graph, mesh = _robot_assets(robot)
    scene = visualize_best_layout(
        layout, evaluator=ev, mounting_graph=graph, mesh=mesh,
        show_rays=True, show_arrows=True, return_scene=True,
    )
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{robot}_policy_bo{n_samples}.glb"
        scene.export(path)
        print(f"  -> saved {path}  (true fitness={fit:.4f}, {len(layout)} sensors)")
    return scene, fit


if __name__ == "__main__":
    import sys

    robots = sys.argv[1:] or shapes.robot_names()
    print(f"Rendering policy best-of-32 layouts for {len(robots)} robots -> {OUT}/")
    for r in robots:
        print(f"[{r}]")
        render_robot(r)
