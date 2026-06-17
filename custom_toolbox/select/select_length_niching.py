"""Length-niching survivor selection for the variable-length sensor-layout GA.

A DEAP-flavoured port of Matt Ryerkerk's metameric selection operator
(``metameric/selection/Selection_LengthNiching.m``, dissertation Ch. 7). The
idea: in a variable-length GA, solutions of different *lengths* (here, sensor
counts) otherwise compete in a single selection pool, and one length tends to
take over the whole population ("length convergence"). Shorter layouts are
usually easier to optimise and out-compete longer ones before the longer ones
have been refined -- so the best achievable layout at each sensor count never
gets a protected, actively-evolving sub-population.

Length niching fixes this structurally: partition the combined parent+child pool
by ``len(individual)`` (sensor count), give each length a quota of the next
generation, and run local selection *within* each length independently. Every
sensor count keeps its own breeding room.

Composition (mirrors the MATLAB decomposition), as configured for this study:

  * **Window** -- which lengths form niches. A *fixed* window over the bounded
    range ``[1, MAX_SENSORS_PER_INDIVIDUAL]`` (``WindowFunction_Fixed``): the
    right choice here because the sensor count is small and hard-bounded, so the
    moving/biased windows (for unbounded, unknown optimal lengths) aren't needed.
    Lengths are ordered by proximity to the current best length (ties -> shorter,
    the ``-0.1`` rule) so any non-divisible remainder of the quota goes to the
    most promising lengths.
  * **Niche size** (``_determine_niche_size``) -- split ``k`` as evenly as
    possible across the niches, remainder to the priority lengths
    (``DetermineNicheSize.m``).
  * **Local selection** -- *per-length elitism* plus a fixed-size DEAP
    tournament. Each non-empty niche carries its single fittest individual
    through unchanged (consuming one of its slots), then fills the rest by
    ``tools.selTournament`` over the *other* candidates.

Choices that differ from the MATLAB original, by design (see the conversation
that specified this port):

  * **Fixed tournament size** (``tournsize``, default 3) rather than Ryerkerk's
    adaptive size -- matches the project's existing ``selTournament(tournsize=3)``
    and keeps a single tunable knob.
  * **Per-length elite** carried unchanged in each niche, to guarantee the best
    layout at every sensor count survives (these are exactly the champions the
    per-length visualisations report).
  * **No penalty / constraint field.** These individuals carry only a (maximised)
    scalar fitness, so Deb's feasible-over-infeasible adjustment is dropped;
    comparison is straight DEAP fitness via ``fitness.wvalues``.
  * **(mu+lambda) survivor selection.** Like the original, this expects the
    *combined* parent+child pool and returns the next generation, so it replaces
    both the elitism and ``selTournament`` steps. Returned individuals are
    references into the input pool -- the evolution loop must clone before
    varying (standard DEAP convention), which it does at the reproduction step.

Usage (notebook)::

    import custom_toolbox.select.select_length_niching as length_niching
    toolbox.register("select_niching", length_niching.sel_length_niching, tournsize=3)
    ...
    # in the loop, with `offspring` already evaluated:
    population[:] = toolbox.select_niching(population + offspring, params.POPULATION_SIZE)
"""

from __future__ import annotations

from deap import tools

from config import params
from config.params import Individual, Population


def _fitness_key(ind: Individual):
    """Sort/compare key for an individual (higher is better).

    ``fitness.wvalues`` are the weight-signed objective values, so a plain
    ``max`` over this key respects the maximise direction without re-checking the
    weights here.
    """
    return ind.fitness.wvalues


def _best_length(pool: Population) -> int:
    """Length (sensor count) of the fittest individual in the pool.

    Drives the niche ordering so any leftover quota favours the lengths nearest
    the current incumbent (``RecordBest`` + the ``-0.1`` rule in the windows).
    """
    return len(max(pool, key=_fitness_key))


def _ordered_window(best_length: int) -> list[int]:
    """Fixed window ``[1, MAX]`` ordered by proximity to ``best_length``.

    Closest length first (priority for leftover allocations); ties broken toward
    shorter solutions via the ``-0.1`` offset, matching ``WindowFunction_Fixed``.
    """
    lengths = range(1, params.MAX_SENSORS_PER_INDIVIDUAL + 1)
    return sorted(lengths, key=lambda length: abs(length - (best_length - 0.1)))


def _determine_niche_size(k: int, num_niche: int) -> list[int]:
    """Allocate ``k`` selection slots across ``num_niche`` niches.

    Port of ``DetermineNicheSize.m``: as even as possible; the ``k % num_niche``
    leftover slots go to the first niches, which the caller has ordered by
    priority (closest to the best length).
    """
    base = k // num_niche
    sizes = [base] * num_niche
    for i in range(k % num_niche):
        sizes[i] += 1
    return sizes


def _select_from_niche(
    candidates: list[Individual], quota: int, tournsize: int
) -> list[Individual]:
    """Select ``quota`` survivors from one length niche.

    Per-length elitism + fixed-size tournament: the niche's single fittest
    individual is carried through unchanged (it consumes one slot), and the
    remaining ``quota - 1`` slots are filled by ``selTournament`` over the *other*
    candidates. If the niche has no more candidates than its quota, all of them
    survive (the shortfall is redistributed by the caller).
    """
    if quota <= 0 or not candidates:
        return []

    # Fewer candidates than slots: keep them all; caller redistributes the rest.
    if len(candidates) <= quota:
        return list(candidates)

    elite = max(candidates, key=_fitness_key)
    # Exclude the elite (by identity) from the tournament pool so it is not
    # double-counted; fill the remaining slots from the others.
    others = [ind for ind in candidates if ind is not elite]
    rest = tools.selTournament(others, quota - 1, tournsize=tournsize)
    return [elite, *rest]


def sel_length_niching(
    individuals: Population, k: int, tournsize: int = 3
) -> Population:
    """Select ``k`` survivors from ``individuals`` via length niching.

    A DEAP ``select`` operator intended for the *combined* parent+child pool (a
    (mu+lambda) survivor step). Register and call it as::

        toolbox.register("select_niching", sel_length_niching, tournsize=3)
        population[:] = toolbox.select_niching(population + offspring, POP_SIZE)

    All individuals must have a valid fitness. Returned individuals are
    references into ``individuals`` -- clone before any in-place variation.

    1. Bucket the pool by ``len(individual)`` over the fixed window ``[1, MAX]``.
    2. Order lengths by proximity to the current best length and split ``k``
       across them (priority lengths get the remainder).
    3. Per-length elite + tournament within each niche.
    4. Redistribute any shortfall (empty or under-populated lengths) by
       tournament over the not-yet-selected individuals -- the fallback in
       ``Selection_LengthNiching.m``.
    """
    if k <= 0 or not individuals:
        return []

    window = _ordered_window(_best_length(individuals))

    # Bucket by length (only lengths inside the fixed window survive here; the
    # variation operators already keep lengths in [1, MAX]).
    niches: dict[int, list[Individual]] = {length: [] for length in window}
    for ind in individuals:
        bucket = niches.get(len(ind))
        if bucket is not None:
            bucket.append(ind)

    niche_sizes = _determine_niche_size(k, len(window))

    selected: list[Individual] = []
    for length, quota in zip(window, niche_sizes):
        selected.extend(_select_from_niche(niches[length], quota, tournsize))

    # Redistribute any shortfall (under-filled / empty niches) over the leftovers,
    # so the population still reaches k. Identity-based dedup since tournaments
    # may return the same object more than once.
    shortfall = k - len(selected)
    if shortfall > 0:
        chosen = set(map(id, selected))
        leftovers = [ind for ind in individuals if id(ind) not in chosen]
        if leftovers:
            selected.extend(tools.selTournament(leftovers, shortfall, tournsize=tournsize))

    # Guard: never return more than requested (keep the fittest if we somehow do).
    if len(selected) > k:
        selected.sort(key=_fitness_key, reverse=True)
        selected = selected[:k]

    return selected
