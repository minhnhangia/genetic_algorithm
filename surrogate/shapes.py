"""Robot shape-library access: per-robot mounting graph + CoverageEvaluator.

Reads ``data/shapes/manifest.json`` (written by ``generate_shapes.py``) and builds,
for any robot, a :class:`CoverageEvaluator` bound to that robot's chassis mesh and
mounting graph -- so footprints are extracted on the same shared evaluation grid
across the whole fleet (the precondition for fixed-size footprint masks).
"""

from __future__ import annotations

import json
import pathlib
import pickle

import networkx as nx

SHAPES_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "shapes"
MANIFEST_PATH = SHAPES_DIR / "manifest.json"


def load_manifest() -> dict[str, dict]:
    """Robot name -> {mesh_path, graph_path, n_nodes, n_edges}."""
    return json.loads(MANIFEST_PATH.read_text())


def robot_names() -> list[str]:
    return list(load_manifest().keys())


def load_graph(name: str) -> nx.Graph:
    with open(load_manifest()[name]["graph_path"], "rb") as f:
        return pickle.load(f)


def build_evaluator(name: str, **evaluator_kwargs):
    """Return ``(evaluator, graph)`` for one robot.

    The evaluator uses the robot's chassis mesh (self-occlusion) and its mounting
    graph (sensor origins); extra kwargs pass through to ``CoverageEvaluator`` but
    the grid params must stay at their defaults so masks remain comparable.
    """
    from custom_toolbox.evaluate.evaluate_fitness_raycast import CoverageEvaluator

    info = load_manifest()[name]
    graph = load_graph(name)
    evaluator = CoverageEvaluator(
        mesh_path=info["mesh_path"], mounting_graph=graph, **evaluator_kwargs
    )
    return evaluator, graph
