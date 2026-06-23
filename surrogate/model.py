"""PyG footprint surrogate: per-sensor coverage-mask prediction (plan Phase 2).

v2 decoder: a small **CNN with circular-theta padding** that decodes the query
conditioning into the two coverage grids -- ground ``(60, 360)`` and cylinder
``(20, 360)``. This bakes in the structure the v1 dot-product decoder ignored: the
azimuth axis is cyclic (circular padding on width), the radial/height axis is not
(replicate padding on height), and the two surfaces are decoded by separate heads.

Architecture:
* **Shape encoder** -- GraphSAGE over a robot's mounting graph (node features =
  centred ``pos`` + ``normal``) -> per-node embedding. Encoded once per robot.
* **Query conditioning** -- queried node embedding fused with a sensor-type
  embedding + the wrap-safe orientation vector.
* **Grid decoder** -- per-surface CNN heads (circular-theta) -> per-cell logits,
  flattened to match ``footprint_flat`` (ground row-major, then cylinder).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from custom_toolbox.evaluate.evaluate_fitness_raycast import CoverageEvaluator


def default_grid_shapes() -> tuple[tuple[int, int], tuple[int, int]]:
    """``(ground_shape, cyl_shape)`` from a default evaluator (e.g. (60,360),(20,360))."""
    ev = CoverageEvaluator()
    return tuple(ev.ground.grid_shape), tuple(ev.cylinder.grid_shape)


class GridDecoder(nn.Module):
    """Decode a conditioning vector into one ``(H, W)`` logit grid.

    Seeds a small feature map from the conditioning code, upsamples to ``(H, W)``,
    and refines with convs that pad **circularly on W (theta)** and by replication on
    H (radius/height) -- so the azimuth wraps but the radial axis doesn't.
    """

    def __init__(self, cond_dim: int, out_hw: tuple[int, int], ch: int = 64) -> None:
        super().__init__()
        self.H, self.W = out_hw
        self.ch = ch
        self.sH, self.sW = max(4, self.H // 4), max(8, self.W // 8)
        self.seed = nn.Linear(cond_dim, ch * self.sH * self.sW)
        self.conv1 = nn.Conv2d(ch, ch, 3)
        self.conv2 = nn.Conv2d(ch, ch, 3)
        self.head = nn.Conv2d(ch, 1, 3)

    def _cconv(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode="circular")  # wrap azimuth (W)
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")  # radius/height (H)
        return conv(x)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        b = cond.shape[0]
        x = self.seed(cond).view(b, self.ch, self.sH, self.sW)
        x = F.interpolate(
            x, size=(self.H, self.W), mode="bilinear", align_corners=False
        )
        x = F.relu(self._cconv(self.conv1, x))
        x = F.relu(self._cconv(self.conv2, x))
        return self.head_forward(x)

    def head_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._cconv(self.head, x)[:, 0]  # (B, H, W)


class FootprintGNN(nn.Module):
    def __init__(
        self,
        ground_shape: tuple[int, int],
        cyl_shape: tuple[int, int],
        node_in: int = 6,
        hidden: int = 128,
        gnn_layers: int = 3,
        n_sensor_types: int = 3,
        orient_dim: int = 6,
        dec_ch: int = 64,
    ) -> None:
        super().__init__()
        self.ground_shape = tuple(ground_shape)
        self.cyl_shape = tuple(cyl_shape)
        self.n_ground = self.ground_shape[0] * self.ground_shape[1]
        self.n_cyl = self.cyl_shape[0] * self.cyl_shape[1]
        self.n_cells = self.n_ground + self.n_cyl

        dims = [node_in] + [hidden] * gnn_layers
        self.gnn = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1]) for i in range(gnn_layers)]
        )
        self.sensor_emb = nn.Embedding(n_sensor_types + 1, hidden)  # value 1..3
        self.orient_mlp = nn.Linear(orient_dim, hidden)
        self.fuse = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden)
        )
        self.ground_dec = GridDecoder(hidden, self.ground_shape, ch=dec_ch)
        self.cyl_dec = GridDecoder(hidden, self.cyl_shape, ch=dec_ch)

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Per-node shape embeddings ``(N, hidden)`` for one robot graph."""
        h = x
        for conv in self.gnn:
            h = F.relu(conv(h, edge_index))
        return h

    def decode(self, node_emb, query_idx, sensor_type, orient) -> torch.Tensor:
        """Per-cell logits ``(B, n_cells)`` for a batch of queries on one robot.

        Order matches ``footprint_flat``: ground grid (row-major) then cylinder.
        """
        q = node_emb[query_idx]
        cond = self.fuse(
            torch.cat(
                [q, self.sensor_emb(sensor_type), self.orient_mlp(orient)], dim=-1
            )
        )
        g = self.ground_dec(cond).reshape(cond.shape[0], -1)  # (B, n_ground)
        c = self.cyl_dec(cond).reshape(cond.shape[0], -1)  # (B, n_cyl)
        return torch.cat([g, c], dim=1)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-cell BCE plus a soft-Dice term (handles sparse footprints)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(-1)
    dice = 1.0 - (2 * inter + 1.0) / (p.sum(-1) + target.sum(-1) + 1.0)
    return bce + dice.mean()


def load_surrogate(ckpt_path, device: str = "cpu"):
    """Reconstruct a trained surrogate from a checkpoint (grid shapes stored in it)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FootprintGNN(tuple(ckpt["ground_shape"]), tuple(ckpt["cyl_shape"])).to(
        device
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
