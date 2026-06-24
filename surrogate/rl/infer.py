"""Best-of-N inference + verify-and-fallback (removes the stochastic collapse zeros).

The greedy(argmax) rollout occasionally lands a held-out robot in a bad basin
(coverage below break-even -> true_fit 0). Two cheap test-time fixes:

* **best-of-N**: also draw N stochastic rollouts, score every candidate with the
  REAL evaluator, keep the best. N+1 layout raycasts (~cheap) -- still far below a
  per-robot footprint table.
* **verify-and-fallback**: if the policy's best is implausibly low (likely a
  collapse), fall back to greedy-on-surrogate (which never collapses) and keep the
  better. Adaptive, so the fast policy path is used for the healthy majority and the
  fallback's surrogate-table cost is paid only on suspected failures.

Everything is true-verified, so the reported fitness is honest.
"""

from __future__ import annotations

from .bc import greedy_surrogate
from .eval_baselines import spread_nodes
from .evaluate import _layout_from_sel
from .train_ppo import _true_fitness, rollout

FALLBACK_THRESHOLD = 0.05  # policy best below this => suspected collapse -> fallback


def policy_candidates(env, policy, robot, n_samples=16):
    """Greedy rollout + N stochastic rollouts; returns list of layouts (rm set to robot)."""
    cands = [rollout(env, policy, robot, greedy=True)[2]]
    for _ in range(n_samples):
        cands.append(rollout(env, policy, robot, greedy=False)[2])
    return cands


def greedy_surrogate_layout(rm, pool_size=100, seed=0):
    """Greedy-on-surrogate layout on the CURRENT robot (rm already encoded)."""
    pool = spread_nodes(rm.graph, pool_size, seed=seed)
    return _layout_from_sel(greedy_surrogate(rm, pool))


def evaluate_robot(env, policy, robot, n_samples=16, pool_size=100,
                   threshold=FALLBACK_THRESHOLD):
    """True-verified greedy / best-of-N / verify-and-fallback fitness for one robot."""
    cands = policy_candidates(env, policy, robot, n_samples)  # leaves env.rm on robot
    fits = [_true_fitness(robot, lay) for lay in cands]
    greedy = fits[0]
    best_of_n = max(fits)

    fb = None
    if best_of_n < threshold:  # suspected collapse -> pay the fallback cost
        fb_layout = greedy_surrogate_layout(env.rm, pool_size)
        fb = _true_fitness(robot, fb_layout)
    final = max(best_of_n, fb) if fb is not None else best_of_n
    return {"greedy": greedy, "best_of_n": best_of_n, "fallback": fb,
            "final": final, "n_fallback": int(fb is not None)}
