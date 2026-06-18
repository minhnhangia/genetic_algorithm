"""Batch, non-interactive mounting-graph generation for the robot fleet.

Generates one mounting graph per robot chassis STL (skipping the interactive crop;
the automated downward + internal-raycast filters in ``generate_ga_graph`` curate
the surface). Each graph is saved to ``data/shapes/<robot>/mounting_graph.pkl`` and
a manifest records every robot's mesh + graph paths + node/edge counts. This is the
shape library the multi-robot footprint dataset (plan Phase 1) consumes.

Run from the project root: ``python generate_shapes.py``.
"""

from __future__ import annotations

import json
import os
import pathlib

import trimesh

from generate_mounting_graph import generate_ga_graph, save_graph

BASE = pathlib.Path(
    os.path.expanduser(
        "~/genetic_algorithm_ws/ros2_ws/src/robotnik_description/meshes/bases"
    )
)
OUT = pathlib.Path(__file__).parent / "data" / "shapes"

# robot name -> chassis STL (relative to BASE). rbkairos is the original robot.
FLEET = {
    "rbkairos": "rbkairos/rbkairos_chassis.stl",
    "rbrobout": "rbrobout/rbrobout_top_cover.stl",
    "rbsummit_xl": "rbsummit/rbsummit_xl_chassis_simple.stl",
    "rbsummit_steel": "rbsummit_steel/rbsummit_steel_chassis.stl",
    "rbtheron": "rbtheron/theron_base_v4.stl",
    "rbtheron_plus_top": "rbtheron/rbtheron_plus_top_chassis.stl",
    "rbwatcher": "rbwatcher/rbwatcher_chassis.stl",
    "rbvogui_xl": "rbvogui_xl/rbvogui_xl_chassis.stl",
}


def main(point_count: int = 30000, neighbor_radius: float = 0.05) -> None:
    manifest: dict[str, dict] = {}
    for name, rel in FLEET.items():
        mesh_path = BASE / rel
        if not mesh_path.exists():
            print(f"[skip] {name}: mesh not found at {mesh_path}")
            continue

        print(f"\n===== {name} ({mesh_path.name}) =====")
        mesh = trimesh.load_mesh(str(mesh_path))
        graph, _ = generate_ga_graph(
            mesh,
            point_count=point_count,
            neighbor_radius=neighbor_radius,
            curate=False,
        )
        graph_path = OUT / name / "mounting_graph.pkl"
        save_graph(graph, graph_path)

        manifest[name] = {
            "mesh_path": str(mesh_path),
            "graph_path": str(graph_path),
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {OUT / 'manifest.json'} ({len(manifest)} robots).")
    for name, info in manifest.items():
        print(f"  {name:18s} {info['n_nodes']:6d} nodes  {info['n_edges']:7d} edges")


if __name__ == "__main__":
    main()
