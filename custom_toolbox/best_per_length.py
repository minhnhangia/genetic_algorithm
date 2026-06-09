"""Track the best individual seen for each sensor count (genome length).

The DEAP :class:`~deap.tools.HallOfFame` keeps the globally best layouts
regardless of size. For this study we *also* want the best layout at each fixed
sensor count -- e.g. "what is the best achievable 1-sensor layout vs the best
3-sensor layout?" -- to expose the coverage/redundancy gained per added sensor
against its extra cost.

``BestPerLength`` mirrors the slice of the HallOfFame API the evolution loop
relies on (:meth:`update`), so it drops in right next to
``hof.update(population)``. Stored individuals are deep-copied, so later in-place
mutation of the population cannot corrupt the recorded champions, and a layout
that goes extinct in later generations is still retained once it has been seen.

Coverage of all sensor counts is effectively guaranteed by the random
initialisation (which samples 1..``MAX_SENSORS_PER_INDIVIDUAL`` uniformly), but
any count never observed is simply absent from the tracker.
"""

from __future__ import annotations

from copy import deepcopy

from config.params import Individual, Population


class BestPerLength:
    """Best-so-far individual for each genome length (sensor count).

    Keyed by ``len(individual)``. Only individuals with a valid fitness are
    considered, and a stored champion is replaced only by a strictly fitter
    individual of the same length.
    """

    def __init__(self) -> None:
        self._bests: dict[int, Individual] = {}

    def update(self, population: Population) -> None:
        """Fold a population into the per-length record (call every generation)."""
        for ind in population:
            if not ind.fitness.valid:
                continue
            length = len(ind)
            if length == 0:
                continue
            incumbent = self._bests.get(length)
            if incumbent is None or ind.fitness > incumbent.fitness:
                # Deep-copy so subsequent in-place mutation can't corrupt it.
                self._bests[length] = deepcopy(ind)

    @property
    def lengths(self) -> list[int]:
        """Sensor counts recorded so far, in ascending order."""
        return sorted(self._bests)

    @property
    def items(self) -> list[Individual]:
        """The best individuals, ordered by ascending sensor count."""
        return [self._bests[length] for length in self.lengths]

    def __getitem__(self, length: int) -> Individual:
        return self._bests[length]

    def __contains__(self, length: int) -> bool:
        return length in self._bests

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self._bests)

    def clear(self) -> None:
        self._bests.clear()
