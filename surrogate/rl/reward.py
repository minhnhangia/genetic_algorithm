"""Surrogate-backed reward model for the placement RL (plan Phase A).

The RL never raycasts: rewards come from the trained footprint surrogate. A
``RewardModel`` binds to one robot (encodes its graph once), then `step(...)` adds a
sensor, predicts its footprint with the surrogate, ORs it into a running union, and
returns the **marginal fitness** `0.7*dCoverage - 0.3*dCost` (cost exact via
``FitnessScorer``). This is the dense reward the policy optimises.

``reward_fidelity`` is the Phase-A gate: it checks the surrogate's marginal reward
tracks the *true* marginal fitness (from the real ``CoverageEvaluator``) over random
placement sequences. If correlation is low, improve the surrogate before training RL.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch_geometric.utils import to_undirected

from config.params import MAX_SENSORS_PER_INDIVIDUAL, Gene
from config.sensors import SENSOR_CATALOG
from custom_toolbox.evaluate.scoring import FitnessScorer

from .. import shapes
from ..footprints import orientation_features, sample_orientation
from ..model import load_surrogate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = pathlib.Path(__file__).resolve().parents[2] / "data" / "surrogate.pt"

# sensor_type value (1..3) -> Sensor object
SENSOR_BY_TYPE = {s.sensor_type.value: s for s in SENSOR_CATALOG.values()}


class RewardModel:
    """Surrogate-driven coverage reward, bound to one robot at a time."""

    def __init__(
        self,
        ckpt_path: pathlib.Path = CKPT,
        threshold: float = 0.5,
        w_cov: float = 0.7,
        w_cost: float = 0.3,
        max_budget: float = 10000.0,
        device: str = DEVICE,
    ) -> None:
        self.device = device
        self.threshold = threshold
        self.model, _ = load_surrogate(ckpt_path, device)
        self.n_cells = self.model.n_cells
        # Must match the evaluator the surrogate was trained against.
        self.scorer = FitnessScorer(w_cov, w_cost, max_budget, total_cells=self.n_cells)
        self._node_emb: torch.Tensor | None = None
        self.reset_state()

    def set_robot(self, name: str) -> None:
        """Load + encode a robot's graph once; reset the running layout."""
        graph = shapes.load_graph(name)
        n = graph.number_of_nodes()
        pos = np.stack([graph.nodes[i]["pos"] for i in range(n)]).astype(np.float32)
        nrm = np.stack([graph.nodes[i]["normal"] for i in range(n)]).astype(np.float32)
        pos = pos - pos.mean(0, keepdims=True)
        x = torch.tensor(np.concatenate([pos, nrm], axis=1), device=self.device)
        ei = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        ei = to_undirected(ei).to(self.device)
        with torch.no_grad():
            self._node_emb = self.model.encode_graph(x, ei)
        self.graph = graph
        self.n_nodes = n
        self.reset_state()

    def reset_state(self) -> None:
        self.union = np.zeros(self.n_cells, dtype=bool)
        self.layout: list[Gene] = []
        self.cur_fit = 0.0

    @torch.no_grad()
    def predict_mask(
        self, node_id: int, sensor_type: int, pitch: int, roll: int, yaw: int
    ) -> np.ndarray:
        """Surrogate's thresholded footprint for one sensor pose (boolean mask)."""
        sensor = SENSOR_BY_TYPE[sensor_type]
        orient = orientation_features(sensor, pitch, roll, yaw).astype(np.float32)
        logits = self.model.decode(
            self._node_emb,
            torch.tensor([node_id], device=self.device),
            torch.tensor([sensor_type], device=self.device),
            torch.tensor(orient[None], device=self.device),
        )
        return (torch.sigmoid(logits)[0] > self.threshold).cpu().numpy()

    @torch.no_grad()
    def predict_masks_batch(
        self, node_ids, sensor_types, oris, chunk: int = 64
    ) -> np.ndarray:
        """Batched thresholded footprints for many poses on the current robot.

        ``oris`` is a list of ``(pitch, roll, yaw)``. Decoding is chunked because
        the CNN decoder's full-grid feature maps are memory-heavy at large batch.
        Returns ``(B, n_cells)`` boolean.
        """
        orient = np.stack(
            [
                orientation_features(SENSOR_BY_TYPE[t], p, r, y)
                for t, (p, r, y) in zip(sensor_types, oris)
            ]
        ).astype(np.float32)
        out = []
        for s in range(0, len(node_ids), chunk):
            sl = slice(s, s + chunk)
            logits = self.model.decode(
                self._node_emb,
                torch.tensor(node_ids[sl], device=self.device),
                torch.tensor(sensor_types[sl], device=self.device),
                torch.tensor(orient[sl], device=self.device),
            )
            out.append((torch.sigmoid(logits) > self.threshold).cpu().numpy())
        return np.concatenate(out, axis=0)

    def terminal_reward(self) -> float:
        """Dense (unclamped) layout fitness for RL: ``w_cov*cov_frac - w_cost*cost_frac``.

        Unlike ``FitnessScorer.score`` (clamped at 0), this stays graded for poor /
        cost-heavy layouts so the terminal RL signal has gradient off the near-zero
        floor. It matches the true fitness wherever that is positive.
        """
        s = self.scorer
        cov_frac = int(self.union.sum()) / s.total_cells
        cost = sum(g.sensor.price for g in self.layout)
        cost_frac = min(cost / s.max_budget, 1.0)
        return s.w_cov * cov_frac - s.w_cost * cost_frac

    def step(
        self, node_id: int, sensor_type: int, pitch: int, roll: int, yaw: int
    ) -> float:
        """Add a sensor; return marginal fitness (commits to running state)."""
        mask = self.predict_mask(node_id, sensor_type, pitch, roll, yaw)
        new_union = self.union | mask
        gene = Gene(
            sensor=SENSOR_BY_TYPE[sensor_type],
            node_id=node_id,
            pitch=pitch,
            roll=roll,
            yaw=yaw,
        )
        new_fit = self.scorer.score(int(new_union.sum()), self.layout + [gene])
        reward = new_fit - self.cur_fit
        self.union, self.cur_fit = new_union, new_fit
        self.layout.append(gene)
        return reward


def reward_fidelity(
    robots: list[str] | None = None, n_sequences: int = 40, seed: int = 0
) -> dict:
    """Phase-A gate: surrogate marginal reward vs TRUE marginal fitness.

    Builds random placement sequences per robot; for each placement records the
    surrogate reward and the true marginal fitness (from the real evaluator).
    Reports Pearson correlation and MAE -- high correlation => the surrogate reward
    is a sound RL signal.
    """
    rm = RewardModel()
    rng = np.random.default_rng(seed)
    robots = robots or shapes.robot_names()
    sur, tru = [], []

    for name in robots:
        rm.set_robot(name)
        evaluator, graph = shapes.build_evaluator(name)
        nodes = np.array(list(graph.nodes()))
        for _ in range(n_sequences):
            rm.reset_state()
            true_layout: list[Gene] = []
            true_prev = 0.0
            k = int(rng.integers(1, MAX_SENSORS_PER_INDIVIDUAL + 1))
            for node in rng.choice(nodes, size=k, replace=False):
                stype = int(rng.integers(1, 4))
                sensor = SENSOR_BY_TYPE[stype]
                p, r, y = sample_orientation(sensor, rng)
                sur.append(rm.step(int(node), stype, p, r, y))
                true_layout.append(
                    Gene(sensor=sensor, node_id=int(node), pitch=p, roll=r, yaw=y)
                )
                true_fit = evaluator.evaluate_individual(true_layout)[0]
                tru.append(true_fit - true_prev)
                true_prev = true_fit

    sur, tru = np.asarray(sur), np.asarray(tru)
    corr = float(np.corrcoef(sur, tru)[0, 1])
    return {
        "n": len(sur),
        "pearson_r": corr,
        "mae": float(np.abs(sur - tru).mean()),
        "sur_mean": float(sur.mean()),
        "tru_mean": float(tru.mean()),
    }


if __name__ == "__main__":
    r = reward_fidelity()
    print(f"reward fidelity over {r['n']} placements:")
    print(f"  Pearson r = {r['pearson_r']:.3f}   MAE = {r['mae']:.5f}")
    print(
        f"  mean surrogate reward = {r['sur_mean']:.5f}  "
        f"mean true reward = {r['tru_mean']:.5f}"
    )
    print(
        f"  verdict: {'OK for RL' if r['pearson_r'] > 0.8 else 'WEAK - improve surrogate first'}"
    )
