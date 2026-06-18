"""PyG footprint surrogate: per-sensor coverage-mask prediction (plan Phase 2).

Architecture:

* **Shape encoder** -- a GraphSAGE GNN over a robot's mounting graph (node features
  = centred ``pos`` + ``normal``) produces a shape-aware embedding per candidate
  node. Encoded once per robot, reused for all its samples.
* **Query conditioning** -- the queried node's embedding is fused with a sensor-type
  embedding and the wrap-safe orientation vector into a conditioning code.
* **Cell decoder** -- each of the 28,800 grid cells has a fixed geometry embedding
  (built from ``(r|z, sin theta, cos theta, surface)`` so the cyclic azimuth and the
  ground/cylinder split are baked in). A cell's logit is the dot product of its
  embedding with the conditioning code, plus a per-cell bias. This is efficient
  (``B x n_cells x d``) and naturally per-cell.

Trained with per-cell BCE (+ Dice) against the true footprint masks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from custom_toolbox.evaluate.evaluate_fitness_raycast import CoverageEvaluator


def build_cell_coords(evaluator: CoverageEvaluator | None = None) -> np.ndarray:
    """Fixed ``(n_cells, 5)`` geometry features per grid cell.

    Order matches ``footprint_flat`` (ground cells row-major, then cylinder):
    ``[coord_norm, sin theta, cos theta, is_ground, is_cyl]`` where ``coord_norm``
    is normalised radius (ground) or height (cylinder).
    """
    evaluator = evaluator or CoverageEvaluator()
    g, c = evaluator.ground, evaluator.cylinder

    rows = []
    # Ground: (n_r, n_az), row-major.
    for ir in range(g.grid_shape[0]):
        r = g.r_min + (ir + 0.5) * g.r_res
        r_norm = (r - g.r_min) / (g.max_radius - g.r_min)
        for ith in range(g.n_az):
            th = (ith + 0.5) * g.dtheta
            rows.append((r_norm, np.sin(th), np.cos(th), 1.0, 0.0))
    # Cylinder: (nz, n_az), row-major.
    z_span = c.nz * c.z_res
    for iz in range(c.grid_shape[0]):
        z = c.z_min + (iz + 0.5) * c.z_res
        z_norm = (z - c.z_min) / z_span
        for ith in range(c.n_az):
            th = (ith + 0.5) * c.dtheta
            rows.append((z_norm, np.sin(th), np.cos(th), 0.0, 1.0))
    return np.asarray(rows, dtype=np.float32)


class FootprintGNN(nn.Module):
    def __init__(
        self,
        cell_coords: np.ndarray,
        node_in: int = 6,
        hidden: int = 128,
        gnn_layers: int = 3,
        n_sensor_types: int = 3,
        orient_dim: int = 6,
    ) -> None:
        super().__init__()
        dims = [node_in] + [hidden] * gnn_layers
        self.gnn = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(gnn_layers)]
        )
        self.sensor_emb = nn.Embedding(n_sensor_types + 1, hidden)  # value 1..3
        self.orient_mlp = nn.Linear(orient_dim, hidden)
        self.fuse = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden)
        )
        self.cell_mlp = nn.Sequential(
            nn.Linear(cell_coords.shape[1], hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.cell_bias = nn.Parameter(torch.zeros(cell_coords.shape[0]))
        self.register_buffer("cell_coords", torch.from_numpy(cell_coords))

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Per-node shape embeddings ``(N, hidden)`` for one robot graph."""
        h = x
        for conv in self.gnn:
            h = F.relu(conv(h, edge_index))
        return h

    def decode(
        self, node_emb: torch.Tensor, query_idx, sensor_type, orient
    ) -> torch.Tensor:
        """Per-cell logits ``(B, n_cells)`` for a batch of queries on one robot."""
        q = node_emb[query_idx]                              # (B, hidden)
        cond = self.fuse(
            torch.cat([q, self.sensor_emb(sensor_type), self.orient_mlp(orient)], dim=-1)
        )                                                    # (B, hidden)
        cell_emb = self.cell_mlp(self.cell_coords)           # (n_cells, hidden)
        return cond @ cell_emb.t() + self.cell_bias          # (B, n_cells)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-cell BCE plus a soft-Dice term (handles sparse footprints)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(-1)
    dice = 1.0 - (2 * inter + 1.0) / (p.sum(-1) + target.sum(-1) + 1.0)
    return bce + dice.mean()
