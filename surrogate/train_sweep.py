"""Autonomous surrogate-improvement sweep (hold-out-by-robot, early-stopped).

Trains a few principled variants to convergence (early stopping on val IoU + LR
decay + best-checkpoint save), under a global wall-clock deadline so a SELECTED model
is ready within the time budget. Promotes the best variant to ``data/surrogate.pt``
only if it beats the current IoU (0.47) on the same hold-out; otherwise leaves the
current model untouched. Writes an auditable RESULTS.md.

Variants: convergence (current arch, just trained properly), scale-norm inputs
(OOD generalisation), higher capacity, and capacity+scale-norm. The feature contract
(incl. scale-norm) travels with each checkpoint so the RL loads it consistently.

Run:  PYTHONUNBUFFERED=1 python -m surrogate.train_sweep
"""

from __future__ import annotations

import copy
import json
import pathlib
import shutil
import time

import numpy as np
import torch
from torch_geometric.utils import to_undirected

from . import shapes
from .dataset import DEFAULT_OUT as DATA_DIR
from .dataset import load_shard
from .features import build_node_features
from .model import FootprintGNN, dice_bce_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "surrogate_variants"
SURROGATE_PT = pathlib.Path(__file__).resolve().parent.parent / "data" / "surrogate.pt"
CURRENT_IOU = 0.47  # baseline to beat (data/surrogate_baseline_iou047.pt)

GLOBAL_BUDGET_H = 11.0  # leave margin within the 15h window
PER_VARIANT_CAP_H = 3.0
MAX_EPOCHS = 250
PATIENCE = 14  # epochs without val-IoU improvement -> stop


class RobotData:
    def __init__(self, name: str, n_cells: int, scale_norm: bool):
        graph = shapes.load_graph(name)
        feats = build_node_features(graph, scale_norm)
        self.x = torch.tensor(feats, device=DEVICE)
        ei = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        self.edge_index = to_undirected(ei).to(DEVICE)
        shard = load_shard(DATA_DIR / f"{name}.npz")
        self.node_id = torch.tensor(shard["node_id"], device=DEVICE)
        self.sensor_type = torch.tensor(shard["sensor_type"], device=DEVICE)
        self.orient = torch.tensor(shard["orient"], device=DEVICE)
        self.mask_packed = shard["mask_packed"]
        self.n_cells = n_cells

    def __len__(self):
        return len(self.node_id)

    def masks(self, idx):
        bits = np.unpackbits(self.mask_packed[idx], axis=-1)[:, : self.n_cells]
        return torch.tensor(bits, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def evaluate(model, robots):
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
    return {
        "iou": inter / max(union, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "cov_frac_mae": cov_abs / cov_n / robots[0].n_cells,
    }


def train_variant(
    name,
    arch,
    scale_norm,
    train_robots,
    val_robots,
    n_cells,
    ground_shape,
    cyl_shape,
    deadline,
    batch_size=256,
    lr=1e-3,
    seed=0,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FootprintGNN(ground_shape, cyl_shape, **arch).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5
    )
    train_data = [RobotData(r, n_cells, scale_norm) for r in train_robots]
    val_data = [RobotData(r, n_cells, scale_norm) for r in val_robots]

    best = {"iou": -1.0}
    best_state = None
    no_improve = 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        if time.time() > deadline:
            print(f"  [{name}] hit deadline at epoch {epoch}", flush=True)
            break
        model.train()
        total, nb = 0.0, 0
        for rd in train_data:
            order = rng.permutation(len(rd))
            for s in range(0, len(rd), batch_size):
                idx = order[s : s + batch_size]
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
        sched.step(val["iou"])
        dt = (time.time() - t0) / (epoch + 1)
        improved = val["iou"] > best["iou"] + 1e-4
        if improved:
            best = {**val, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"  [{name}] epoch {epoch:3d}  loss={total/nb:.4f}  "
            f"val IoU={val['iou']:.4f} F1={val['f1']:.4f} MAE={val['cov_frac_mae']:.4f}  "
            f"best={best['iou']:.4f}  {dt:.1f}s/ep",
            flush=True,
        )
        if no_improve >= PATIENCE:
            print(
                f"  [{name}] early stop (no val-IoU gain {PATIENCE} epochs)", flush=True
            )
            break

    OUT.mkdir(exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "arch": arch,
        "scale_norm": scale_norm,
        "ground_shape": ground_shape,
        "cyl_shape": cyl_shape,
        "val_robots": val_robots,
        "metrics": best,
    }
    torch.save(ckpt, OUT / f"{name}.pt")
    print(
        f"  [{name}] saved -> {OUT / (name + '.pt')}  best IoU={best['iou']:.4f}\n",
        flush=True,
    )
    return best


def main():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    n_cells = manifest["n_cells"]
    ground_shape = tuple(manifest["ground_shape"])
    cyl_shape = tuple(manifest["cyl_shape"])
    all_robots = list(manifest["robots"].keys())
    val_robots = all_robots[-2:]  # rbwatcher, rbvogui_xl -- matches RL hold-out
    train_robots = [r for r in all_robots if r not in val_robots]

    cap = {"gnn_layers": 4, "hidden": 192, "dec_ch": 96}
    variants = [
        ("convergence", {}, False),
        ("scale_norm", {}, True),
        ("capacity", cap, False),
        ("capacity_scalenorm", cap, True),
    ]

    start = time.time()
    global_deadline = start + GLOBAL_BUDGET_H * 3600
    print(
        f"SWEEP start  train={train_robots}  val={val_robots}  device={DEVICE}",
        flush=True,
    )
    print(f"budget {GLOBAL_BUDGET_H}h, baseline IoU={CURRENT_IOU}\n", flush=True)

    results = {}
    for name, arch, sn in variants:
        if time.time() > global_deadline:
            print(f"global deadline reached; skipping {name}", flush=True)
            continue
        deadline = min(global_deadline, time.time() + PER_VARIANT_CAP_H * 3600)
        print(f"=== variant {name}  arch={arch}  scale_norm={sn} ===", flush=True)
        try:
            results[name] = train_variant(
                name,
                arch,
                sn,
                train_robots,
                val_robots,
                n_cells,
                ground_shape,
                cyl_shape,
                deadline,
            )
        except Exception as e:  # one variant failing must not kill the sweep
            print(f"  [{name}] FAILED: {e}", flush=True)
            results[name] = {"iou": -1.0, "error": str(e)}
        _write_results(results, start)

    # Selection + safe promotion
    ranked = sorted(results.items(), key=lambda kv: kv[1].get("iou", -1), reverse=True)
    best_name, best_m = ranked[0]
    promoted = False
    if best_m.get("iou", -1) > CURRENT_IOU:
        shutil.copy(SURROGATE_PT, SURROGATE_PT.with_suffix(".pt.prev"))
        shutil.copy(OUT / f"{best_name}.pt", SURROGATE_PT)
        promoted = True
    _write_results(results, start, best_name=best_name, promoted=promoted)
    print(
        f"\nBEST: {best_name} IoU={best_m.get('iou'):.4f}  "
        f"{'PROMOTED to surrogate.pt' if promoted else 'NOT promoted (<= 0.47); kept current'}",
        flush=True,
    )


def _write_results(results, start, best_name=None, promoted=False):
    lines = [
        "# Surrogate sweep results",
        "",
        f"elapsed: {(time.time()-start)/3600:.2f} h | baseline IoU {CURRENT_IOU}",
        "",
        "| variant | val IoU | F1 | cov_frac_MAE | best epoch |",
        "|---|---|---|---|---|",
    ]
    for name, m in sorted(
        results.items(), key=lambda kv: kv[1].get("iou", -1), reverse=True
    ):
        lines.append(
            f"| {name} | {m.get('iou',-1):.4f} | {m.get('f1',float('nan')):.4f} "
            f"| {m.get('cov_frac_mae',float('nan')):.4f} | {m.get('epoch','-')} |"
        )
    if best_name:
        verdict = (
            "PROMOTED to surrogate.pt"
            if promoted
            else "kept current (no variant beat 0.47)"
        )
        lines += ["", f"**Best: {best_name} — {verdict}**"]
    OUT.mkdir(exist_ok=True)
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
