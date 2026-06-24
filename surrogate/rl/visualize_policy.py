"""Visualize the sensor layout + coverage the POLICY chooses on a (fleet) robot.

Reuses the notebook helpers in utils/visualization.py:
  * ``visualize_coverage_maps`` -- 2D ground (polar r,theta) + cylinder (theta,z)
    occupancy maps; works with any evaluator, so the fleet evaluator drops in.
  * ``visualize_best_layout``  -- 3D mesh + sensor bodies + rays + coverage points;
    now fleet-capable via its ``mounting_graph`` / ``mesh`` args.

The layout is obtained by best-of-N (true-verified) -- exactly what we report -- so
the picture matches the headline metric. In a notebook call ``show_policy_choice(...)``;
headless, call ``save_policy_choice(...)`` to write a coverage-map PNG + a 3D .glb.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
import trimesh

from utils.visualization import visualize_best_layout, visualize_coverage_maps

from .. import shapes
from .env import PlacementEnv
from .infer import policy_candidates
from .policy import PlacementPolicy
from .reward import RewardModel
from .train_ppo import _true_fitness

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "viz"


def _mesh_for(robot):
    info = json.loads((shapes.MANIFEST_PATH).read_text())[robot]
    return trimesh.load(info["mesh_path"], force="mesh")


def policy_layout(env, policy, robot, n_samples=16):
    """Best-of-N (true-verified) layout the policy picks on ``robot``."""
    cands = policy_candidates(env, policy, robot, n_samples)
    fits = [_true_fitness(robot, lay) for lay in cands]
    i = int(np.argmax(fits))
    return cands[i], float(fits[i])


def _prepare(robot, policy=None, ckpt=None, n_samples=16):
    if policy is None:
        policy = PlacementPolicy().to(DEVICE)
        if ckpt is not None:
            policy.load_state_dict(torch.load(ckpt, map_location=DEVICE)["state_dict"])
    env = PlacementEnv(RewardModel(device=DEVICE))
    layout, fit = policy_layout(env, policy, robot, n_samples)
    evaluator, graph = shapes.build_evaluator(robot)
    return layout, fit, evaluator, graph


def show_policy_choice(
    robot, policy=None, ckpt=None, n_samples=16, show_3d=True, **layout_kwargs
):
    """Notebook: render the policy's chosen layout (2D maps + optional 3D viewer)."""
    layout, fit, evaluator, graph = _prepare(robot, policy, ckpt, n_samples)
    print(f"=== {robot}: policy best-of-{n_samples} layout, true fitness={fit:.4f} ===")
    visualize_coverage_maps(layout, evaluator=evaluator)
    if show_3d:
        visualize_best_layout(
            layout,
            evaluator=evaluator,
            mounting_graph=graph,
            mesh=_mesh_for(robot),
            **layout_kwargs,
        )
    return layout, fit


def save_policy_choice(robot, policy=None, ckpt=None, n_samples=16, out_dir=OUT):
    """Headless: write a coverage-map PNG + a 3D .glb of the policy's chosen layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout, fit, evaluator, graph = _prepare(robot, policy, ckpt, n_samples)
    print(
        f"{robot}: best-of-{n_samples} true fitness={fit:.4f}  (n={len(layout)} sensors)"
    )

    png = out_dir / f"{robot}_coverage.png"
    _orig = plt.show
    plt.show = lambda *a, **k: plt.gcf().savefig(png, dpi=130, bbox_inches="tight")
    try:
        visualize_coverage_maps(layout, evaluator=evaluator)
    finally:
        plt.show = _orig

    scene = visualize_best_layout(
        layout,
        evaluator=evaluator,
        mounting_graph=graph,
        mesh=_mesh_for(robot),
        return_scene=False,
    )
    glb = out_dir / f"{robot}_layout.glb"
    scene.export(glb)
    print(f"  saved {png.name} + {glb.name} -> {out_dir}")
    return png, glb


def save_comparison(
    robot, policy=None, ckpt=None, n_samples=16, n_starts=40, out_dir=OUT
):
    """Side-by-side coverage maps: POLICY best-of-N vs the multi-start+LS OPTIMUM."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .verify_ceiling import optimum_layout

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_p, fit_p, ev, _ = _prepare(robot, policy, ckpt, n_samples)
    layout_o, _ = optimum_layout(robot, n_starts=n_starts)
    fit_o = ev.evaluate_individual(layout_o)[0]
    dp, do = ev.coverage_debug(layout_p), ev.coverage_debug(layout_o)

    ground, cyl = ev.ground, ev.cylinder
    r_edges = ground.r_min + np.arange(ground.n_r + 1) * ground.r_res
    th_edges = np.arange(ground.n_az + 1) * ground.dtheta
    z_max = cyl.z_min + cyl.nz * cyl.z_res
    Th, Rr = np.meshgrid(th_edges, r_edges)

    fig = plt.figure(figsize=(13, 11))

    def panel(row, debug, label, fit):
        g = debug["ground_grid"].astype(float)
        c = debug["cyl_grid"].astype(float)
        ax1 = fig.add_subplot(2, 2, row * 2 + 1, projection="polar")
        ax1.pcolormesh(Th, Rr, g, cmap="Greens", vmin=0, vmax=1, shading="flat")
        ax1.set_rmin(0.0)
        ax1.set_title(f"{label}\nground {g.mean():.1%}", fontsize=11)
        ax2 = fig.add_subplot(2, 2, row * 2 + 2)
        ax2.imshow(
            c,
            origin="lower",
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=1,
            extent=[0.0, 360.0, cyl.z_min, z_max],
        )
        ax2.set_xlabel("azimuth θ (deg)")
        ax2.set_ylabel("height z (m)")
        ax2.set_title(
            f"{label}  wall {c.mean():.1%}   (fitness {fit:.4f})", fontsize=11
        )

    panel(0, dp, f"POLICY  best-of-{n_samples}", fit_p)
    panel(1, do, "OPTIMUM  multi-start+LS", fit_o)
    fig.suptitle(
        f"{robot}:  policy {fit_p:.4f}  vs  optimum {fit_o:.4f}   "
        f"(policy = {100 * fit_p / fit_o:.0f}% of optimum)",
        fontsize=13,
    )
    fig.tight_layout()
    png = out_dir / f"{robot}_comparison.png"
    fig.savefig(str(png), dpi=130, bbox_inches="tight")
    print(
        f"{robot}: policy {fit_p:.4f} vs optimum {fit_o:.4f}  ({100*fit_p/fit_o:.0f}%) -> {png.name}"
    )
    return png


if __name__ == "__main__":
    import sys

    robot = sys.argv[1] if len(sys.argv) > 1 else "rbwatcher"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    save_policy_choice(robot, ckpt=ckpt)
