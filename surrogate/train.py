"""Train the footprint surrogate with hold-out-by-robot validation (plan Phase 2).

Each robot's mounting graph is encoded by the GNN; its samples are decoded to
footprint masks and supervised with per-cell BCE + Dice. Validation robots are
held out entirely (never seen in training) so the metrics measure real cross-robot
transfer, not interpolation. Reported: per-cell IoU/F1 and coverage-fraction MAE.

Run: ``python -m surrogate.train`` (defaults hold out 2 robots).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch_geometric.utils import to_undirected

from .dataset import DEFAULT_OUT as DATA_DIR
from .dataset import load_shard
from .model import FootprintGNN, dice_bce_loss
from . import shapes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RobotData:
    """One robot's GNN inputs + its footprint samples (masks kept bit-packed)."""

    def __init__(self, name: str, n_cells: int):
        graph = shapes.load_graph(name)
        n = graph.number_of_nodes()
        pos = np.stack([graph.nodes[i]["pos"] for i in range(n)]).astype(np.float32)
        nrm = np.stack([graph.nodes[i]["normal"] for i in range(n)]).astype(np.float32)
        pos = pos - pos.mean(0, keepdims=True)  # centre -> translation invariance
        self.x = torch.tensor(np.concatenate([pos, nrm], axis=1), device=DEVICE)
        ei = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        self.edge_index = to_undirected(ei).to(DEVICE)

        shard = load_shard(DATA_DIR / f"{name}.npz")
        self.node_id = torch.tensor(shard["node_id"], device=DEVICE)
        self.sensor_type = torch.tensor(shard["sensor_type"], device=DEVICE)
        self.orient = torch.tensor(shard["orient"], device=DEVICE)
        self.mask_packed = shard["mask_packed"]  # numpy uint8 (S, 3600)
        self.n_cells = n_cells

    def __len__(self) -> int:
        return len(self.node_id)

    def masks(self, idx: np.ndarray) -> torch.Tensor:
        bits = np.unpackbits(self.mask_packed[idx], axis=-1)[:, : self.n_cells]
        return torch.tensor(bits, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def evaluate(model, robots: list[RobotData]) -> dict:
    model.eval()
    inter = union = tp = fp = fn = 0
    cov_abs = cov_n = 0.0
    for rd in robots:
        node_emb = model.encode_graph(rd.x, rd.edge_index)
        for s in range(0, len(rd), 512):
            idx = np.arange(s, min(s + 512, len(rd)))
            logits = model.decode(
                node_emb, rd.node_id[idx], rd.sensor_type[idx], rd.orient[idx]
            )
            pred = (torch.sigmoid(logits) > 0.5).float()
            tgt = rd.masks(idx)
            inter += (pred * tgt).sum().item()
            union += ((pred + tgt) > 0).float().sum().item()
            tp += (pred * tgt).sum().item()
            fp += (pred * (1 - tgt)).sum().item()
            fn += ((1 - pred) * tgt).sum().item()
            cov_abs += (pred.sum(-1) - tgt.sum(-1)).abs().sum().item()
            cov_n += len(idx)
    iou = inter / max(union, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"iou": iou, "f1": f1, "cov_frac_mae": cov_abs / cov_n / robots[0].n_cells}


def train(
    val_robots: list[str] | None = None,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    n_cells = manifest["n_cells"]
    all_robots = list(manifest["robots"].keys())
    val_robots = val_robots or all_robots[-2:]
    train_robots = [r for r in all_robots if r not in val_robots]
    print(f"train: {train_robots}\nval (held out): {val_robots}\ndevice: {DEVICE}\n")

    ground_shape = tuple(manifest["ground_shape"])
    cyl_shape = tuple(manifest["cyl_shape"])
    model = FootprintGNN(ground_shape, cyl_shape).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = [RobotData(r, n_cells) for r in train_robots]
    val_data = [RobotData(r, n_cells) for r in val_robots]
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        total, nb = 0.0, 0
        for rd in train_data:
            order = rng.permutation(len(rd))
            for s in range(0, len(rd), batch_size):
                idx = order[s : s + batch_size]
                # Re-encode per step so gradients flow through the GNN each update.
                node_emb = model.encode_graph(rd.x, rd.edge_index)
                logits = model.decode(
                    node_emb, rd.node_id[idx], rd.sensor_type[idx], rd.orient[idx]
                )
                loss = dice_bce_loss(logits, rd.masks(idx))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
        val = evaluate(model, val_data)
        print(
            f"epoch {epoch:2d}  train_loss={total/nb:.4f}  "
            f"val IoU={val['iou']:.3f}  F1={val['f1']:.3f}  "
            f"cov_frac_MAE={val['cov_frac_mae']:.4f}"
        )

    ckpt = pathlib.Path(__file__).resolve().parent.parent / "data" / "surrogate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "val_robots": val_robots,
            "ground_shape": ground_shape,
            "cyl_shape": cyl_shape,
        },
        ckpt,
    )
    print(f"\nsaved {ckpt}")


if __name__ == "__main__":
    train()
