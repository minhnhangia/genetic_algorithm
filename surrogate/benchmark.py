"""Single-robot benchmark: greedy-on-true-footprints vs the GA.

Both layouts are scored by the TRUE evaluator for a fair comparison. This is the
current-robot sanity check and the upper-bound reference for the (future) learned
surrogate; the held-out multi-robot benchmark with confidence intervals (plan
Phase 4) follows once the surrogate exists. Run as ``python -m surrogate.benchmark``.
"""

from __future__ import annotations

import random
import time

from deap import base, creator, tools

import config.params as params
import custom_toolbox.evaluate.evaluate_fitness_raycast as ev
import custom_toolbox.initialize.initialize as initialize
import custom_toolbox.mate.mate_spatial as mate_spatial
import custom_toolbox.mutate.mutate as mutate
from utils.ga_run import TOURNAMENT, run_evolution

from .select import optimize_layout


def _fmt(layout) -> str:
    return ", ".join(
        f"{g.sensor.sensor_type.name}@{g.node_id}(p{g.pitch},r{g.roll},y{g.yaw})"
        for g in layout
    )


def main(
    n_nodes: int = 80, ga_pop: int = 5000, ga_ngen: int = 15, seed: int = 0
) -> None:
    evaluator = ev.CoverageEvaluator()

    # --- Greedy on true footprints ---
    t = time.time()
    layout, greedy_est = optimize_layout(
        evaluator, n_candidate_nodes=n_nodes, seed=seed
    )
    greedy_time = time.time() - t
    greedy_true = evaluator.evaluate_individual(layout)[0]

    # --- GA baseline (modest budget; wall times reported for context) ---
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    tb = base.Toolbox()
    tb.register("individual", initialize.create_individual, creator.Individual)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", mate_spatial.cx_spatial_front_back)
    tb.register("mutate", mutate.mutate_sensor_layout)
    tb.register("evaluate", evaluator.evaluate_individual)
    tb.register("select", tools.selTournament, tournsize=3)

    random.seed(seed)
    pop = tb.population(n=ga_pop)
    t = time.time()
    res = run_evolution(
        pop,
        tb,
        strategy=TOURNAMENT,
        seed=seed,
        population_size=ga_pop,
        ngen=ga_ngen,
        verbose=False,
    )
    ga_time = time.time() - t
    ga_fit = res.hof[0].fitness.values[0]

    print("\n================ greedy(true footprints) vs GA ================")
    print(
        f"GREEDY  fit={greedy_true:.5f} ({len(layout)} sensors)  "
        f"est={greedy_est:.5f}  time={greedy_time:.1f}s"
    )
    print(f"        {_fmt(layout)}")
    print(
        f"GA      fit={ga_fit:.5f} ({len(res.hof[0])} sensors)  "
        f"time={ga_time:.1f}s  (pop={ga_pop}, ngen={ga_ngen})"
    )
    print(
        f"\ngreedy/GA fitness ratio: {greedy_true / ga_fit:.3f}  "
        f"| greedy speedup: {ga_time / max(greedy_time, 1e-9):.1f}x"
    )
    print(f"union-approx check |est-true|={abs(greedy_est - greedy_true):.5f}")


if __name__ == "__main__":
    main()
