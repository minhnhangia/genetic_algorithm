"""Phase 1: coverage-aware CONSTRUCTIVE policy (isolated; reuses existing primitives).

The original policy is coverage-blind (obs = node_emb + used_mask + n_placed) -> it applies
one fixed template and under-covers the ground. Per the literature (S2V-DQN/RELS-DQN), the
missing feature is per-candidate MARGINAL coverage gain. Here:

* `CoverageEnv` operates over a fixed POOL = spread_nodes(graph, 100, seed 0) -- the SAME pool as
  `cached_true_table`, so the greedy-on-true teacher, the surrogate cache, and the action space
  all align. On reset it builds the surrogate footprint cache (`reward.build_candidate_cache`).
  Action.node is a POOL INDEX; the env maps it to a graph node. Obs carries `coverage_frac` and
  per-pool-node `node_gains` (from the cache, vs the running union).
* `CoveragePolicy` mirrors `PlacementPolicy`'s act/evaluate interface (so `train_ppo.rollout`,
  `ppo_update`, and `infer` best-of-N reuse unchanged) but feeds `node_gains` into the node head
  and `coverage_frac` into the context.

Everything new lives here -> zero risk to the existing env/policy/checkpoints.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .bc_true import cached_true_table
from .env import ORIENT_BINS, N_ORIENT, N_TYPES, Action, Obs, PlacementEnv
from .eval_baselines import spread_nodes
from .reward import RewardModel

POOL_SIZE = 100  # must match cached_true_table's pool (spread_nodes(g, 100, seed=0))


class CoverageEnv(PlacementEnv):
    """Pool-based, coverage-aware variant of PlacementEnv (same step/reset interface)."""

    def __init__(self, reward_model: RewardModel, pool_size: int = POOL_SIZE,
                 true_gain: bool = False, **kw):
        super().__init__(reward_model, **kw)
        self.pool_size = pool_size
        # true_gain: the marginal-gain FEATURE comes from the EXACT raycast footprint
        # table (cached_true_table) instead of the surrogate decoder. Reward is still
        # the true raycaster either way -- this only de-noises the state feature.
        self.true_gain = true_gain
        self._cached_robot: str | None = None
        self._true_memo: dict = {}  # robot -> (pool, masks_gpu, cn, ct, co); avoid disk reload

    def _load_true_cache(self, robot_name: str) -> None:
        """Populate rm's candidate cache from the cached TRUE footprint table (seed-0 pool,
        which matches spread_nodes(g,100,0)). Ordering is node-major (build_true_table),
        identical to build_candidate_cache, so node_marginal_gains' view(P, n_poses) aligns."""
        if robot_name not in self._true_memo:
            masks, cn, ct, co = cached_true_table(robot_name, self.pool_size, seed=0)
            pool = cn.reshape(-1, 3 * len(ORIENT_BINS))[:, 0].copy()  # first pose per node
            masks_t = torch.as_tensor(masks, device=self.rm.device)  # bool (M, n_cells)
            self._true_memo[robot_name] = (pool, masks_t, cn, ct, co)
        pool, masks_t, cn, ct, co = self._true_memo[robot_name]
        self.pool = pool
        self.rm.pool = np.asarray(pool)
        self.rm.n_poses = 3 * len(ORIENT_BINS)
        self.rm.cand_masks = masks_t
        self.rm.cand_node, self.rm.cand_type, self.rm.cand_orient = cn, ct, co

    def reset(self, robot_name: str) -> Obs:
        # Re-encode + rebuild the cache only when the robot changes (best-of-N resets
        # the same robot many times -> avoid redundant GNN encode + decode).
        if robot_name != self._cached_robot:
            self.rm.set_robot(robot_name)  # GNN encode (once per robot)
            if self.true_gain:
                self._load_true_cache(robot_name)  # exact-gain feature (no surrogate decode)
            else:
                self.pool = spread_nodes(self.rm.graph, self.pool_size, seed=0)
                self.rm.build_candidate_cache(self.pool)  # surrogate footprint cache
            self._pool_t = torch.as_tensor(self.pool, device=self.rm.device, dtype=torch.long)
            self._cached_robot = robot_name
        else:
            self.rm.reset_state()  # clear union/layout, keep encode + cache
        self.robot = robot_name
        self.P = len(self.pool)
        self.used = torch.zeros(self.P, dtype=torch.bool, device=self.rm.device)
        self.n_placed = 0
        return self._obs()

    def _obs(self) -> Obs:
        return Obs(
            node_emb=self.rm._node_emb[self._pool_t].detach(),  # (P, H)
            used_mask=self.used.clone(),
            n_placed=self.n_placed,
            coverage_frac=float(self.rm.union.sum()) / self.rm.n_cells,
            node_gains=self.rm.node_marginal_gains(self.rm.union).detach(),  # (P,)
        )

    def step(self, action: Action):
        if action.stop or self.n_placed >= self.max_sensors:
            return self._obs(), self._terminal_reward(), True, {"layout": list(self.rm.layout)}
        node = int(self.pool[action.node])  # pool index -> graph node id
        pitch, roll, yaw = ORIENT_BINS[action.orient]
        self.rm.step(node, action.sensor_type, pitch, roll, yaw)  # updates union + layout
        self.used[action.node] = True
        self.n_placed += 1
        done = self.n_placed >= self.max_sensors
        reward = self._terminal_reward() if done else 0.0
        info = {"layout": list(self.rm.layout)} if done else {}
        return self._obs(), reward, done, info


class CoveragePolicy(nn.Module):
    """Actor-critic with coverage_frac in the context and node_gains in the node head."""

    def __init__(self, node_dim: int = 128, step_dim: int = 16, hid: int = 64) -> None:
        super().__init__()
        self.step_emb = nn.Embedding(8, step_dim)
        self.type_emb = nn.Embedding(N_TYPES, step_dim)
        ctx_dim = node_dim + step_dim + 1  # pooled graph + step + coverage_frac
        self.node_head = nn.Sequential(  # +1 for per-node marginal gain
            nn.Linear(node_dim + ctx_dim + 1, hid), nn.ReLU(), nn.Linear(hid, 1)
        )
        self.stop_head = nn.Sequential(nn.Linear(ctx_dim, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.type_head = nn.Sequential(
            nn.Linear(node_dim + ctx_dim, hid), nn.ReLU(), nn.Linear(hid, N_TYPES)
        )
        self.orient_head = nn.Sequential(
            nn.Linear(node_dim + step_dim + ctx_dim, hid), nn.ReLU(), nn.Linear(hid, N_ORIENT)
        )
        self.value_head = nn.Sequential(nn.Linear(ctx_dim, hid), nn.ReLU(), nn.Linear(hid, 1))

    def _ctx(self, obs: Obs):
        H = obs.node_emb  # (P, node_dim)
        step = self.step_emb(torch.tensor(min(obs.n_placed, 7), device=H.device))
        cov = torch.tensor([obs.coverage_frac], device=H.device, dtype=H.dtype)
        ctx = torch.cat([H.mean(0), step, cov])  # (ctx_dim,)
        return H, ctx, obs.node_gains

    def _place_logits(self, H, ctx, gains, used, allow_stop=True):
        ctx_b = ctx.unsqueeze(0).expand(H.shape[0], -1)
        node_in = torch.cat([H, ctx_b, gains.unsqueeze(-1)], -1)
        node_logits = self.node_head(node_in).squeeze(-1).masked_fill(used, float("-inf"))
        stop_logit = self.stop_head(ctx) if allow_stop else ctx.new_full((1,), float("-inf"))
        return torch.cat([node_logits, stop_logit])

    def value(self, obs: Obs):
        _, ctx, _ = self._ctx(obs)
        return self.value_head(ctx).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: Obs, greedy: bool = False):
        H, ctx, gains = self._ctx(obs)
        place = Categorical(logits=self._place_logits(H, ctx, gains, obs.used_mask, obs.n_placed > 0))
        a = place.probs.argmax() if greedy else place.sample()
        n = H.shape[0]
        if int(a) == n:
            return Action(stop=True), float(place.log_prob(a)), float(self.value(obs)), (int(a), -1, -1)
        td = Categorical(logits=self.type_head(torch.cat([H[a], ctx])))
        t = td.probs.argmax() if greedy else td.sample()
        od = Categorical(logits=self.orient_head(torch.cat([H[a], self.type_emb(t), ctx])))
        o = od.probs.argmax() if greedy else od.sample()
        logp = place.log_prob(a) + td.log_prob(t) + od.log_prob(o)
        act = Action(stop=False, node=int(a), sensor_type=int(t) + 1, orient=int(o))
        return act, float(logp), float(self.value(obs)), (int(a), int(t), int(o))

    def evaluate(self, obs: Obs, raw_idx):
        a_idx, t_idx, o_idx = raw_idx
        H, ctx, gains = self._ctx(obs)
        place = Categorical(logits=self._place_logits(H, ctx, gains, obs.used_mask, obs.n_placed > 0))
        a = torch.tensor(a_idx, device=H.device)
        value = self.value_head(ctx).squeeze(-1)
        if a_idx == H.shape[0]:
            return place.log_prob(a), place.entropy(), value
        td = Categorical(logits=self.type_head(torch.cat([H[a_idx], ctx])))
        t = torch.tensor(t_idx, device=H.device)
        od = Categorical(logits=self.orient_head(torch.cat([H[a_idx], self.type_emb(t), ctx])))
        o = torch.tensor(o_idx, device=H.device)
        logp = place.log_prob(a) + td.log_prob(t) + od.log_prob(o)
        ent = place.entropy() + td.entropy() + od.entropy()
        return logp, ent, value


# ---------------------------------------------------------------------------
# Coverage-aware BC (greedy-on-true teacher, pool-index targets) + training
# ---------------------------------------------------------------------------
import torch.nn.functional as F  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_NPOSE = 3 * N_ORIENT


def _cache_row(pool_idx, tdx0, oi):
    return pool_idx * _NPOSE + tdx0 * N_ORIENT + oi


def generate_coverage_demos(robots, n_episodes=30, subset=40, seed=0):
    """Greedy-on-TRUE teacher over the shared pool; demos carry the ordered prefix of
    (pool_idx, type_idx0, orient) so the coverage Obs can be reconstructed at pretrain."""
    from .bc_true import cached_true_table, greedy_over_masks
    from .reward import SENSOR_BY_TYPE
    from custom_toolbox.evaluate.scoring import FitnessScorer

    rng = np.random.default_rng(seed)
    demos = []
    for robot in robots:
        masks, cn, ct, co = cached_true_table(robot)  # TRUE footprints, pool 100, node-major
        prices = np.array([SENSOR_BY_TYPE[int(t)].price for t in ct], dtype=float)
        scorer = FitnessScorer(0.7, 0.3, 10000.0, total_cells=masks.shape[1])
        pool = list(dict.fromkeys(int(x) for x in cn))  # pool order (node-major preserves it)
        node_to_idx = {n: i for i, n in enumerate(pool)}
        for _ in range(n_episodes):
            sub = set(int(x) for x in rng.choice(pool, size=min(subset, len(pool)), replace=False))
            sel = greedy_over_masks(masks, cn, ct, co, scorer, sub)  # [(node, tdx0, oi)]
            prefix = []
            for node, tdx0, oi in sel:
                pidx = node_to_idx[node]
                demos.append((robot, tuple(prefix), ("place", pidx, tdx0, oi)))
                prefix.append((pidx, tdx0, oi))
            demos.append((robot, tuple(prefix), ("stop",)))
    return demos


def _demo_obs(rm, pool_t, prefix):
    """Reconstruct the coverage Obs for a demo prefix using the SURROGATE cache."""
    union = torch.zeros(rm.n_cells, dtype=torch.bool, device=rm.device)
    used = torch.zeros(len(pool_t), dtype=torch.bool, device=rm.device)
    for pidx, tdx0, oi in prefix:
        union |= rm.cand_masks[_cache_row(pidx, tdx0, oi)]
        used[pidx] = True
    return Obs(
        node_emb=rm._node_emb[pool_t].detach(),
        used_mask=used,
        n_placed=len(prefix),
        coverage_frac=float(union.sum()) / rm.n_cells,
        node_gains=rm.node_marginal_gains(union).detach(),
    )


def coverage_bc_pretrain(policy, demos, epochs=12, lr=1e-3, seed=0, true_gain=False):
    """Clone the greedy-on-true teacher's (node|STOP, type, orient) decisions, in the
    coverage Obs the policy will see at inference. ``true_gain`` makes the cloned obs use
    the EXACT raycast cache (so BC matches the true-gain PPO/eval obs distribution)."""
    rm = RewardModel(device=DEVICE)
    rm.n_poses = 3 * len(ORIENT_BINS)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    cache: dict = {}

    def prep(robot):
        if robot not in cache:
            rm.set_robot(robot)
            if true_gain:
                masks, cn, _, _ = cached_true_table(robot, POOL_SIZE, seed=0)
                pool = cn.reshape(-1, rm.n_poses)[:, 0].copy()
                rm.cand_masks = torch.as_tensor(masks, device=DEVICE)
            else:
                pool = spread_nodes(rm.graph, POOL_SIZE, seed=0)
                rm.build_candidate_cache(pool)
            rm.pool = np.asarray(pool)
            cache[robot] = (rm._node_emb.detach().clone(),
                            torch.as_tensor(pool, device=DEVICE, dtype=torch.long),
                            rm.cand_masks.clone())
        return cache[robot]

    for ep in range(epochs):
        rng.shuffle(demos)
        tot, nb = 0.0, 0
        for robot, prefix, target in demos:
            emb, pool_t, masks = prep(robot)
            rm._node_emb, rm.cand_masks = emb, masks  # point rm at this robot's cache
            obs = _demo_obs(rm, pool_t, prefix)
            H, ctx, gains = policy._ctx(obs)
            place = policy._place_logits(H, ctx, gains, obs.used_mask, obs.n_placed > 0).unsqueeze(0)
            P = H.shape[0]
            if target[0] == "stop":
                loss = F.cross_entropy(place, torch.tensor([P], device=DEVICE))
            else:
                _, pidx, tdx0, oi = target
                tl = policy.type_head(torch.cat([H[pidx], ctx])).unsqueeze(0)
                te = policy.type_emb(torch.tensor(tdx0, device=DEVICE))
                ol = policy.orient_head(torch.cat([H[pidx], te, ctx])).unsqueeze(0)
                loss = (F.cross_entropy(place, torch.tensor([pidx], device=DEVICE))
                        + F.cross_entropy(tl, torch.tensor([tdx0], device=DEVICE))
                        + F.cross_entropy(ol, torch.tensor([oi], device=DEVICE)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"  cov-BC epoch {ep:2d}  loss={tot/nb:.4f}", flush=True)
    return policy


def train_coverage(val_robots, init_policy, iters=50, episodes_per_iter=48,
                   lr=3e-4, gamma=0.99, seed=0, eval_every=50, true_reward=True,
                   single_robot=None, true_gain=False):
    """PPO over the fleet with hold-out, reusing train_ppo.rollout + ppo_update."""
    import random
    from .. import shapes
    from .train_ppo import _true_fitness, ppo_update, rollout

    torch.manual_seed(seed); random.seed(seed)
    env = CoverageEnv(RewardModel(device=DEVICE), true_reward=true_reward, true_gain=true_gain)
    policy = init_policy
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    all_robots = shapes.robot_names()
    if single_robot:  # overfit gate
        train_robots, val_robots = [single_robot], [single_robot]
    else:
        train_robots = [r for r in all_robots if r not in val_robots]
    for it in range(iters):
        batch, ep_rets = [], []
        for _ in range(episodes_per_iter):
            robot = random.choice(train_robots)
            trans, term_r, _ = rollout(env, policy, robot)
            T = len(trans)
            for k, (obs, raw, lp, val) in enumerate(trans):
                ret = (gamma ** (T - 1 - k)) * term_r
                batch.append((obs, raw, ret, ret - val, lp))
            ep_rets.append(term_r)
        loss = ppo_update(policy, opt, batch)
        if it % eval_every == 0 or it == iters - 1:
            ev = []
            for r in val_robots:
                _, _, lay = rollout(env, policy, r, greedy=True)
                ev.append(f"{r}:{_true_fitness(r, lay):.4f}(n{len(lay)})")
            print(f"iter {it:3d} train_return={np.mean(ep_rets):+.4f} loss={loss:.3f} | {'  '.join(ev)}", flush=True)
    return policy


def cov_run_fold(held, all_robots, seed=0, iters=50, true_gain=False):
    from .infer import evaluate_robot

    train_robots = [r for r in all_robots if r not in held]
    print(f"\n=== cov fold: held-out {held} (seed {seed}) "
          f"[{'TRUE-gain' if true_gain else 'surrogate-gain'}] ===", flush=True)
    demos = generate_coverage_demos(train_robots, seed=seed)
    pol = CoveragePolicy().to(DEVICE)
    coverage_bc_pretrain(pol, demos, epochs=12, seed=seed, true_gain=true_gain)
    pol = train_coverage(val_robots=held, init_policy=pol, true_reward=True,
                         iters=iters, eval_every=iters, seed=seed, true_gain=true_gain)
    env = CoverageEnv(RewardModel(device=DEVICE), true_gain=true_gain)
    out = {}
    for r in held:
        m = evaluate_robot(env, pol, r)
        out[r] = m
        print(f"  held-out {r:18s} greedy={m['greedy']:.4f} bestN={m['best_of_n']:.4f} "
              f"final={m['final']:.4f}", flush=True)
    return out


if __name__ == "__main__":
    import os
    import sys

    from .. import shapes
    from .kfold import _multiseed_summary, optimum_fit

    # The TRUE-raycast marginal-gain fork: the gain FEATURE is exact (cached_true_table)
    # instead of surrogate-decoded. Reward is the true raycaster either way.
    TRUE_GAIN = os.environ.get("COV_TRUE_GAIN") == "1"
    tag = "TRUE-gain" if TRUE_GAIN else "surrogate-gain"

    mode = sys.argv[1] if len(sys.argv) > 1 else "kfold"
    if mode == "overfit":  # cheap gate: true-fit must climb before any 5h k-fold
        robot = sys.argv[2] if len(sys.argv) > 2 else "rbkairos"
        print(f"=== COVERAGE overfit gate ({robot}) [{tag}] ===", flush=True)
        demos = generate_coverage_demos([robot], n_episodes=30)
        pol = CoveragePolicy().to(DEVICE)
        coverage_bc_pretrain(pol, demos, epochs=12, true_gain=TRUE_GAIN)
        train_coverage(val_robots=[robot], init_policy=pol, single_robot=robot,
                       true_reward=True, iters=30, eval_every=3, true_gain=TRUE_GAIN)
    else:  # multi-seed k-fold vs OPT
        seeds = [int(s) for s in (sys.argv[2:] or ["0", "1", "2"])]
        robots = shapes.robot_names()
        opt = {r: optimum_fit(r) for r in robots}
        folds = [robots[i : i + 2] for i in range(0, len(robots), 2)]
        per_seed = {}
        for seed in seeds:
            print(f"\n########## COVERAGE SEED {seed} [{tag}] ##########", flush=True)
            res = {}
            for held in folds:
                res.update(cov_run_fold(held, robots, seed=seed, true_gain=TRUE_GAIN))
            per_seed[seed] = res
        _multiseed_summary(per_seed, opt, seeds)
