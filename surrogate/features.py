"""Shared GNN node-feature builder.

Used by BOTH surrogate training and RL inference (reward.py) so the input contract
is identical -- critical when a scale-normalised surrogate is promoted: the RL must
normalise inputs the same way or the frozen embeddings are garbage. The scale-norm
flag travels with the model (``model.scale_norm``).
"""

from __future__ import annotations

import numpy as np


def build_node_features(graph, scale_norm: bool = False) -> np.ndarray:
    """Return ``(N, 6)`` features = centred position (3) + surface normal (3).

    With ``scale_norm`` the centred positions are divided by the RMS radius (a single
    scalar, so aspect ratio is preserved), making differently-sized chassis
    comparable -- targets cross-robot OOD generalisation. Normals are already unit.
    """
    n = graph.number_of_nodes()
    pos = np.stack([graph.nodes[i]["pos"] for i in range(n)]).astype(np.float32)
    nrm = np.stack([graph.nodes[i]["normal"] for i in range(n)]).astype(np.float32)
    pos = pos - pos.mean(0, keepdims=True)  # centre -> translation invariance
    if scale_norm:
        rms = float(np.sqrt((pos**2).sum(1).mean())) + 1e-6
        pos = pos / rms
    return np.concatenate([pos, nrm], axis=1)
