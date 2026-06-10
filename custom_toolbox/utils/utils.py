"""Shared helpers for the GA toolbox.

Currently hosts :func:`select_spread_nodes`, used both when initializing an
individual and when the Add mutation places a new sensor, so the spread-out
placement logic lives in exactly one place.
"""

import random
from collections.abc import Iterable

from config import params
from config.graph import MOUNTING_GRAPH

# Mounting-node positions as plain (x, y, z) tuples, aligned with
# params.VALID_NODE_IDS, plus the inverse node_id -> index lookup. Both are
# precomputed once. Tuples (rather than a NumPy array) are deliberate: the
# distance check below runs over at most a handful of points, where NumPy's
# per-call overhead (alloc, sqrt, ufunc dispatch) dwarfs the arithmetic --
# pure-Python floats with an early-exit benchmark ~7x faster than np.linalg.norm.
_NODE_POSITIONS: list[tuple[float, float, float]] = [
    (
        float(MOUNTING_GRAPH.nodes[i]["pos"][0]),
        float(MOUNTING_GRAPH.nodes[i]["pos"][1]),
        float(MOUNTING_GRAPH.nodes[i]["pos"][2]),
    )
    for i in params.VALID_NODE_IDS
]
_NODE_INDEX: dict[int, int] = {
    node_id: i for i, node_id in enumerate(params.VALID_NODE_IDS)
}


def select_spread_nodes(
    num_nodes: int,
    min_separation_m: float,
    existing_nodes: Iterable[int] = (),
) -> list[int]:
    """Pick ``num_nodes`` distinct mounting nodes that are spatially spread out.

    Poisson-disk style rejection sampling: each chosen node must lie at least
    ``min_separation_m`` (Euclidean) from every other chosen node *and* from
    every node in ``existing_nodes`` -- already-placed sensors that act as
    constraints but are not returned. This is shared by individual creation
    (no existing nodes) and the Add mutation (one new node clear of the sensors
    already in the layout). If the threshold can't be met within an attempt
    budget it is progressively relaxed (and ultimately drops to 0), so this
    always returns ``num_nodes`` distinct, previously-unused node IDs and never
    deadlocks -- even for large separations or on dense/small graphs.

    Args:
        num_nodes: how many new nodes to choose and return.
        min_separation_m: minimum Euclidean distance kept between any two of the
            relevant nodes (chosen and existing alike).
        existing_nodes: node IDs already occupied; picks avoid and stay clear of
            them, but they are not included in the result.

    Returns:
        Up to ``num_nodes`` distinct node IDs not in ``existing_nodes`` (fewer
        only if the graph has too few free nodes to satisfy the count).
    """
    positions = _NODE_POSITIONS
    n = len(positions)
    max_attempts = 32

    # Seed the working set with the already-placed nodes as pure constraints;
    # only the picks made after this offset are returned.
    chosen: list[int] = [_NODE_INDEX[node_id] for node_id in existing_nodes]
    start = len(chosen)
    num_nodes = min(num_nodes, n - start)

    while len(chosen) - start < num_nodes:
        threshold = min_separation_m
        # Positions of the nodes already committed -- the constraints this pick
        # must clear. Recomputed only when a node is appended (once per pick),
        # not on every attempt.
        chosen_pos = [positions[i] for i in chosen]
        while True:
            thr_sq = threshold * threshold
            for _ in range(max_attempts):
                cand = random.randrange(n)

                # Skip candidates already accepted or already occupied.
                if cand in chosen:
                    continue

                # Very first node with no constraints: accept without checks.
                if not chosen_pos:
                    chosen.append(cand)
                    break

                # Enforce the minimum separation: squared distances against the
                # squared threshold (no sqrt), bailing on the first violation.
                px, py, pz = positions[cand]
                far_enough = True
                for qx, qy, qz in chosen_pos:
                    dx = px - qx
                    dy = py - qy
                    dz = pz - qz
                    if dx * dx + dy * dy + dz * dz < thr_sq:
                        far_enough = False
                        break
                if far_enough:
                    chosen.append(cand)
                    break
            else:
                # No candidate satisfied the threshold within the budget.
                if threshold > 0.0:
                    threshold = threshold * 0.5 if threshold > 1e-3 else 0.0
                    continue
                # threshold == 0: accept any unused node (one is guaranteed).
                while True:
                    cand = random.randrange(n)
                    if cand not in chosen:
                        chosen.append(cand)
                        break
            break

    return [params.VALID_NODE_IDS[i] for i in chosen[start:]]
