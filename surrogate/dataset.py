"""Multi-robot single-sensor footprint dataset generator (plan Phase 1).

For each robot in the shape library, sample ``(node, sensor_type, orientation)``
triples and extract the 28,800-cell footprint mask via the per-robot evaluator.
Footprints are stored bit-packed (3,600 bytes each); per-sample features are the
node ``pos``/``normal``, sensor type, and the wrap-safe orientation vector. The full
robot graph (the GNN's shape context) is NOT duplicated per sample -- it is loaded
from the shape library at train time via :mod:`surrogate.shapes`.

Candidate nodes are subsampled per robot (the graphs have up to ~22k nodes, far
more than needed) to keep generation and training tractable.

Run: ``python -m surrogate.dataset`` (smoke) or import :func:`generate`.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from config.sensors import SENSOR_CATALOG

from . import shapes
from .footprints import (
    footprint_flat,
    orientation_features,
    sample_orientation,
    sensor_footprint,
)

DEFAULT_OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "footprints"


def generate(
    robots: list[str] | None = None,
    nodes_per_robot: int = 200,
    orientations_per_sensor: int = 4,
    out_dir: pathlib.Path = DEFAULT_OUT,
    seed: int = 0,
) -> dict:
    """Generate per-robot footprint shards and a dataset manifest.

    Returns the manifest dict. Each robot's ``.npz`` holds, for ``S`` samples:
    ``node_id (S,)``, ``node_pos (S,3)``, ``node_normal (S,3)``,
    ``sensor_type (S,)``, ``orient (S,6)``, ``mask_packed (S, 3600) uint8``.
    """
    robots = robots or shapes.robot_names()
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sensors = list(SENSOR_CATALOG.values())

    manifest: dict = {"robots": {}, "params": {
        "nodes_per_robot": nodes_per_robot,
        "orientations_per_sensor": orientations_per_sensor,
        "seed": seed,
    }}

    for name in robots:
        evaluator, graph = shapes.build_evaluator(name)
        manifest.setdefault("n_cells", evaluator._scorer.total_cells)
        manifest.setdefault("ground_shape", list(evaluator.ground.grid_shape))
        manifest.setdefault("cyl_shape", list(evaluator.cylinder.grid_shape))

        all_nodes = np.array(list(graph.nodes()))
        k = min(nodes_per_robot, len(all_nodes))
        chosen = rng.choice(all_nodes, size=k, replace=False)

        node_id, node_pos, node_norm, s_type, orient, masks = [], [], [], [], [], []
        for node in chosen:
            node = int(node)
            pos = np.asarray(graph.nodes[node]["pos"], dtype=np.float32)
            nrm = np.asarray(graph.nodes[node]["normal"], dtype=np.float32)
            for sensor in sensors:
                for _ in range(orientations_per_sensor):
                    p, r, y = sample_orientation(sensor, rng)
                    gr, cy = sensor_footprint(evaluator, sensor, node, p, r, y)
                    node_id.append(node)
                    node_pos.append(pos)
                    node_norm.append(nrm)
                    s_type.append(sensor.sensor_type.value)
                    orient.append(orientation_features(sensor, p, r, y))
                    masks.append(np.packbits(footprint_flat(gr, cy)))

        shard = out_dir / f"{name}.npz"
        np.savez_compressed(
            shard,
            node_id=np.asarray(node_id, dtype=np.int64),
            node_pos=np.asarray(node_pos, dtype=np.float32),
            node_normal=np.asarray(node_norm, dtype=np.float32),
            sensor_type=np.asarray(s_type, dtype=np.int64),
            orient=np.asarray(orient, dtype=np.float32),
            mask_packed=np.asarray(masks, dtype=np.uint8),
        )
        manifest["robots"][name] = {"shard": str(shard), "n_samples": len(node_id)}
        print(f"  {name:18s} {len(node_id):6d} samples -> {shard.name}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def unpack_masks(mask_packed: np.ndarray, n_cells: int) -> np.ndarray:
    """Unpack ``(S, 3600) uint8`` back to ``(S, n_cells)`` boolean footprints."""
    return np.unpackbits(mask_packed, axis=-1)[:, :n_cells].astype(bool)


def load_shard(path: str | pathlib.Path) -> dict:
    """Load one robot's ``.npz`` shard as a dict of arrays."""
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


if __name__ == "__main__":
    # Smoke: one robot, a handful of nodes/orientations.
    m = generate(
        robots=["rbvogui_xl"],
        nodes_per_robot=10,
        orientations_per_sensor=2,
        out_dir=DEFAULT_OUT / "_smoke",
    )
    print("smoke manifest:", json.dumps(m, indent=2))
