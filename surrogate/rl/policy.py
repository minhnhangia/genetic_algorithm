"""Factorized actor-critic over the surrogate's frozen node embeddings (plan Phase C).

The policy does NOT re-encode the graph: it consumes the surrogate's frozen
per-node embeddings (the shared-perception design) and adds light heads:

* a **pointer** over nodes + a **STOP** logit (place a sensor here, or end),
* conditioned **sensor-type** and **orientation-bin** heads,
* a **value** head.

The action is sampled factorized: (node | STOP) -> type -> orientation. Used nodes
are masked. ``act`` samples for rollouts; ``evaluate`` recomputes log-prob/entropy/
value for PPO updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import N_ORIENT, N_TYPES, Action, Obs


class PlacementPolicy(nn.Module):
    def __init__(self, node_dim: int = 128, step_dim: int = 16, hid: int = 64) -> None:
        super().__init__()
        self.step_emb = nn.Embedding(8, step_dim)  # n_placed index
        self.type_emb = nn.Embedding(N_TYPES, step_dim)
        ctx_dim = node_dim + step_dim  # pooled graph + step
        self.node_head = nn.Sequential(
            nn.Linear(node_dim + ctx_dim, hid), nn.ReLU(), nn.Linear(hid, 1)
        )
        self.stop_head = nn.Sequential(
            nn.Linear(ctx_dim, hid), nn.ReLU(), nn.Linear(hid, 1)
        )
        self.type_head = nn.Sequential(
            nn.Linear(node_dim + ctx_dim, hid), nn.ReLU(), nn.Linear(hid, N_TYPES)
        )
        self.orient_head = nn.Sequential(
            nn.Linear(node_dim + step_dim + ctx_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, N_ORIENT),
        )
        self.value_head = nn.Sequential(
            nn.Linear(ctx_dim, hid), nn.ReLU(), nn.Linear(hid, 1)
        )

    def _ctx(self, obs: Obs) -> tuple[torch.Tensor, torch.Tensor]:
        H = obs.node_emb  # (N, node_dim)
        step = self.step_emb(torch.tensor(min(obs.n_placed, 7), device=H.device))
        ctx = torch.cat([H.mean(0), step])  # (ctx_dim,)
        return H, ctx

    def _place_logits(
        self,
        H: torch.Tensor,
        ctx: torch.Tensor,
        used: torch.Tensor,
        allow_stop: bool = True,
    ):
        ctx_b = ctx.unsqueeze(0).expand(H.shape[0], -1)
        node_logits = self.node_head(torch.cat([H, ctx_b], -1)).squeeze(-1)
        node_logits = node_logits.masked_fill(used, float("-inf"))
        # A zero-sensor layout is never valid -> forbid STOP before any placement.
        stop_logit = (
            self.stop_head(ctx) if allow_stop else ctx.new_full((1,), float("-inf"))
        )
        return torch.cat([node_logits, stop_logit])  # (N+1,), last = STOP

    def value(self, obs: Obs) -> torch.Tensor:
        _, ctx = self._ctx(obs)
        return self.value_head(ctx).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: Obs, greedy: bool = False):
        """Sample (or argmax, if ``greedy``) an action; return (Action, logp, value, raw)."""
        H, ctx = self._ctx(obs)
        place = Categorical(
            logits=self._place_logits(
                H, ctx, obs.used_mask, allow_stop=obs.n_placed > 0
            )
        )
        a = place.probs.argmax() if greedy else place.sample()
        n = H.shape[0]
        if int(a) == n:  # STOP
            logp = place.log_prob(a)
            return (
                Action(stop=True),
                float(logp),
                float(self.value(obs)),
                (int(a), -1, -1),
            )

        td = Categorical(logits=self.type_head(torch.cat([H[a], ctx])))
        t = td.probs.argmax() if greedy else td.sample()
        od = Categorical(
            logits=self.orient_head(torch.cat([H[a], self.type_emb(t), ctx]))
        )
        o = od.probs.argmax() if greedy else od.sample()
        logp = place.log_prob(a) + td.log_prob(t) + od.log_prob(o)
        act = Action(stop=False, node=int(a), sensor_type=int(t) + 1, orient=int(o))
        return act, float(logp), float(self.value(obs)), (int(a), int(t), int(o))

    def evaluate(self, obs: Obs, raw_idx: tuple[int, int, int]):
        """Recompute (logp, entropy, value) for a stored transition (for PPO)."""
        a_idx, t_idx, o_idx = raw_idx
        H, ctx = self._ctx(obs)
        place = Categorical(
            logits=self._place_logits(
                H, ctx, obs.used_mask, allow_stop=obs.n_placed > 0
            )
        )
        a = torch.tensor(a_idx, device=H.device)
        value = self.value_head(ctx).squeeze(-1)
        if a_idx == H.shape[0]:  # STOP
            return place.log_prob(a), place.entropy(), value

        td = Categorical(logits=self.type_head(torch.cat([H[a_idx], ctx])))
        t = torch.tensor(t_idx, device=H.device)
        od = Categorical(
            logits=self.orient_head(torch.cat([H[a_idx], self.type_emb(t), ctx]))
        )
        o = torch.tensor(o_idx, device=H.device)
        logp = place.log_prob(a) + td.log_prob(t) + od.log_prob(o)
        ent = place.entropy() + td.entropy() + od.entropy()
        return logp, ent, value
