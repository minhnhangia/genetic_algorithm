"""Sequential sensor-placement environment with a terminal surrogate reward.

An episode builds a layout one sensor at a time on a chosen robot. The reward is
**terminal** -- 0 on every intermediate placement, and the surrogate's predicted
*final layout fitness* at the end (STOP, or the sensor cap). This sidesteps the
noisy per-step marginal (the surrogate ranks whole layouts well, Spearman ~0.86,
even though marginals are hard) -- the design decision behind this RL track.

The policy perceives the robot through the surrogate's **frozen** GNN node
embeddings (exposed in the observation), so the surrogate is the shared perception
for both reward and policy. Episodes are <=`MAX_SENSORS_PER_INDIVIDUAL` steps, so
the sparse terminal signal is easy to credit-assign.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from config.params import MAX_SENSORS_PER_INDIVIDUAL

from .reward import RewardModel

# Fixed orientation action grid (pitch, roll, yaw); roll=0 (redundant for omni,
# minor for directional). Fine pitch grid incl. near-level tilts -- a coarse
# {-30,0,30} capped the achievable coverage ~40% below this on some robots
# (the 120-deg directional sensor needs ~+/-5..15 deg aim). 7 pitches x 6 azimuths.
ORIENT_BINS: list[tuple[int, int, int]] = [
    (p, 0, y)
    for p in (-30, -15, -5, 0, 5, 15, 30)
    for y in (-180, -120, -60, 0, 60, 120)
]
N_ORIENT = len(ORIENT_BINS)
N_TYPES = 3  # sensor_type values 1..3


@dataclass
class Obs:
    node_emb: torch.Tensor  # (N, H) frozen surrogate embeddings
    used_mask: torch.Tensor  # (N,) bool, True where a sensor already sits
    n_placed: int


@dataclass
class Action:
    stop: bool
    node: int = -1  # node index (== node_id, graph is 0..N-1)
    sensor_type: int = 1
    orient: int = 0  # index into ORIENT_BINS


class PlacementEnv:
    def __init__(
        self,
        reward_model: RewardModel,
        max_sensors: int = MAX_SENSORS_PER_INDIVIDUAL,
        true_reward: bool = False,
    ) -> None:
        self.rm = reward_model
        self.max_sensors = max_sensors
        # When True the terminal reward is the real raycast fitness (one
        # evaluate per episode end), which can exceed the surrogate's own
        # ceiling; perception still comes from the surrogate embeddings.
        self.true_reward = true_reward
        self._eval_cache: dict = {}

    def reset(self, robot_name: str) -> Obs:
        self.rm.set_robot(robot_name)  # encodes graph once (frozen) + resets layout
        self.robot = robot_name
        self.n_nodes = self.rm.n_nodes
        self.used = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.rm.device)
        self.n_placed = 0
        return self._obs()

    def _terminal_reward(self) -> float:
        if not self.true_reward:
            # The surrogate's own predicted fitness is the reward
            return self.rm.terminal_reward()
        if not self.rm.layout:
            return 0.0
        from .. import shapes

        # True raycast fitness, but UNCLAMPED (cost-calibrated stopping): the real
        # evaluator clamps fitness at 0, which flattens the reward on robots where a
        # full-cost layout can't clear break-even -> PPO gets no gradient to prefer
        # stopping over over-committing. Recompute w_cov*cov - w_cost*cost without the
        # clamp so each sensor must justify its cost (mirrors RewardModel.terminal_reward).
        if self.robot not in self._eval_cache:
            self._eval_cache[self.robot] = shapes.build_evaluator(self.robot)[0]
        ev = self._eval_cache[self.robot]
        layout = list(self.rm.layout)
        ev.evaluate_individual(layout)  # sets last_ground_grid / last_cyl_grid
        s = ev._scorer
        cov_frac = (int(ev.last_ground_grid.sum()) + int(ev.last_cyl_grid.sum())) / s.total_cells
        cost_frac = min(sum(g.sensor.price for g in layout) / s.max_budget, 1.0)
        return s.w_cov * cov_frac - s.w_cost * cost_frac

    def _obs(self) -> Obs:
        # Clone the mask so stored transitions keep the mask AS OF that step
        # (self.used is mutated in place across the episode).
        return Obs(
            node_emb=self.rm._node_emb.detach(),
            used_mask=self.used.clone(),
            n_placed=self.n_placed,
        )

    def step(self, action: Action) -> tuple[Obs, float, bool, dict]:
        """Apply a placement (or STOP). Reward is terminal = dense surrogate fitness."""
        if action.stop or self.n_placed >= self.max_sensors:
            return (
                self._obs(),
                self._terminal_reward(),
                True,
                {"layout": list(self.rm.layout)},
            )

        pitch, roll, yaw = ORIENT_BINS[action.orient]
        self.rm.step(action.node, action.sensor_type, pitch, roll, yaw)
        self.used[action.node] = True
        self.n_placed += 1

        done = self.n_placed >= self.max_sensors
        reward = self._terminal_reward() if done else 0.0
        info = {"layout": list(self.rm.layout)} if done else {}
        return self._obs(), reward, done, info
