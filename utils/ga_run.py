"""Reusable evolution driver, so a full GA run is a single call.

The notebook's evolution loop is extracted here verbatim (both selection paths)
so it can be invoked more than once -- in particular to run the two selection
strategies back to back for a controlled comparison (see ``utils.comparison``).

``run_evolution`` deep-copies the initial population it is handed and, if given a
``seed``, reseeds the RNG before the loop. Run two strategies from the *same*
initial population and seed and the only difference between them is the selection
operator -- a clean controlled experiment. The expensive fitness evaluator is
deliberately *not* owned here: pass the same ``CoverageEvaluator`` (via
``toolbox.evaluate``) to both runs so its genome cache is shared.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from deap import tools

import config.params as params
from utils.best_per_length import BestPerLength
from utils.per_length_evolution import PerLengthEvolution


@dataclass
class RunResult:
    """Everything one GA run produces, for downstream visualisation."""

    label: str
    logbook: tools.Logbook
    hof: tools.HallOfFame
    best_per_length: BestPerLength
    per_length_evolution: PerLengthEvolution
    population: list


def _make_stats() -> tools.Statistics:
    """Fresh population-fitness statistics (mirrors the notebook's ``stats``)."""
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def run_evolution(
    initial_population: list,
    toolbox: Any,
    *,
    use_length_niching: bool,
    label: str | None = None,
    ngen: int = params.NGEN,
    population_size: int = params.POPULATION_SIZE,
    elite_count: int = params.ELITE_COUNT,
    cx_prob: float = 0.7,
    seed: int | None = None,
    verbose: bool = True,
) -> RunResult:
    """Run one full evolution and return its results.

    Args:
        initial_population: the seeded starting population. Deep-copied here, so
            the caller's list is left untouched and multiple runs are independent.
        toolbox: a DEAP toolbox with ``clone``, ``evaluate``, ``mate``,
            ``mutate``, ``map``, ``select`` (tournament) and ``select_niching``
            registered (exactly the notebook's toolbox).
        use_length_niching: ``True`` -> (mu+lambda) length-niching survivor
            selection; ``False`` -> the original elitism + tournament scheme.
        label: human-readable name for plots; defaults from the strategy.
        ngen / population_size / elite_count / cx_prob: GA settings (default to
            ``config.params``); override for a cheaper comparison run.
        seed: if given, ``random.seed(seed)`` before the loop so runs are
            reproducible and two strategies share an identical RNG start.

    Returns:
        A :class:`RunResult` with the logbook, hall of fame, per-length trackers,
        and the final population.
    """
    if label is None:
        label = "Length niching" if use_length_niching else "Tournament + elitism"

    if seed is not None:
        random.seed(seed)

    # Independent copy so repeated runs don't share (and mutate) individuals.
    population = deepcopy(initial_population)

    # Resolve selection operators. Prefer the toolbox registrations (so a custom
    # tournsize set in the Selection cell applies), but fall back to module
    # defaults so a stale or never-run Selection cell can't break the run.
    select_tournament = getattr(toolbox, "select", None)
    if select_tournament is None:
        select_tournament = lambda pop, k: tools.selTournament(pop, k, tournsize=3)
    select_niching = getattr(toolbox, "select_niching", None)
    if select_niching is None:
        from custom_toolbox.select.select_length_niching import sel_length_niching

        select_niching = sel_length_niching

    stats = _make_stats()
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    hof = tools.HallOfFame(1)
    best_per_length = BestPerLength()
    per_length_evolution = PerLengthEvolution()

    # Initial baseline evaluation.
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(population)
    best_per_length.update(population)

    if verbose:
        strategy = "LENGTH NICHING (mu+lambda)" if use_length_niching else "tournament + elitism"
        print(f"=== Run '{label}'  |  selection: {strategy} ===")

    for gen in range(ngen):
        if use_length_niching:
            # ---- (mu+lambda) with length-niching survivor selection ----
            # Reproduction: every parent breeds once -> lambda = population_size.
            offspring = list(map(toolbox.clone, population))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Survivor selection over the COMBINED parent+child pool.
            population[:] = select_niching(population + offspring, population_size)

        else:
            # ---- Original generational scheme: global elitism + tournament ----
            elites = tools.selBest(population, elite_count)
            elites = list(map(toolbox.clone, elites))

            offspring = select_tournament(population, len(population) - elite_count)
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring + elites

        hof.update(population)
        best_per_length.update(population)

        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        per_length_evolution.record(gen, population)

        if verbose:
            print(logbook.stream)

    if verbose:
        print(
            f"--- '{label}' done: best fitness={hof[0].fitness.values[0]:.4f} "
            f"at {len(hof[0])} sensor(s) ---\n"
        )

    return RunResult(
        label=label,
        logbook=logbook,
        hof=hof,
        best_per_length=best_per_length,
        per_length_evolution=per_length_evolution,
        population=population,
    )
