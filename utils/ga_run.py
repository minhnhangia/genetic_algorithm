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


# Selection strategies recognised by run_evolution.
TOURNAMENT = "tournament"  # global elitism + tournament (the original scheme)
TOURNAMENT_PER_LENGTH_ELITE = "tournament_per_length_elite"  # per-length elitism + tournament
LENGTH_NICHING = "length_niching"  # (mu+lambda) length-niching survivor selection

_STRATEGY_LABELS = {
    TOURNAMENT: "Tournament + global elite",
    TOURNAMENT_PER_LENGTH_ELITE: "Tournament + per-length elite",
    LENGTH_NICHING: "Length niching",
}


def _per_length_elites(population: list, elite_count: int) -> list:
    """The best ``elite_count`` individuals of *each* sensor count (length).

    Contrast with global elitism (``tools.selBest(population, elite_count)``),
    which keeps the ``elite_count`` best overall and so protects only the
    currently-dominant length. This buckets by ``len(individual)`` and keeps each
    length's own best, so every sensor count carries champions forward. A length
    with fewer than ``elite_count`` members contributes all of them. Returns
    references (the caller clones).
    """
    by_length: dict[int, list] = {}
    for ind in population:
        by_length.setdefault(len(ind), []).append(ind)
    elites: list = []
    for members in by_length.values():
        elites.extend(tools.selBest(members, min(elite_count, len(members))))
    return elites


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
    strategy: str = TOURNAMENT,
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
        strategy: which selection scheme to use -- one of ``TOURNAMENT`` (global
            elitism + tournament), ``TOURNAMENT_PER_LENGTH_ELITE`` (per-length
            elitism + tournament), or ``LENGTH_NICHING`` ((mu+lambda) niching).
        label: human-readable name for plots; defaults from the strategy.
        ngen / population_size / elite_count / cx_prob: GA settings (default to
            ``config.params``); override for a cheaper comparison run.
        seed: if given, ``random.seed(seed)`` before the loop so runs are
            reproducible and two strategies share an identical RNG start.

    Returns:
        A :class:`RunResult` with the logbook, hall of fame, per-length trackers,
        and the final population.
    """
    if strategy not in _STRATEGY_LABELS:
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected one of {list(_STRATEGY_LABELS)}."
        )
    if label is None:
        label = _STRATEGY_LABELS[strategy]

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
        print(f"=== Run '{label}'  |  strategy: {strategy} ===")

    for gen in range(ngen):
        if strategy == LENGTH_NICHING:
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
            # ---- Generational scheme: elitism + tournament ----
            # Elitism scope is the only difference between the two tournament
            # strategies: globally best `elite_count`, or `elite_count` per length.
            if strategy == TOURNAMENT_PER_LENGTH_ELITE:
                elites = _per_length_elites(population, elite_count)
            else:
                elites = tools.selBest(population, elite_count)
            elites = list(map(toolbox.clone, elites))

            # Tournament fills the rest globally. Size by the actual elite count
            # (per-length elitism carries up to elite_count * #lengths individuals).
            n_offspring = max(0, len(population) - len(elites))
            offspring = select_tournament(population, n_offspring)
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
