"""Record per-generation fitness statistics grouped by sensor count.

The DEAP :class:`~deap.tools.Logbook` (driven by ``stats``) summarises the whole
population each generation. ``PerLengthEvolution`` parallels it but partitions
each generation's population by genome length, so the evolution of each sensor
count can be plotted as its own curve -- e.g. to see whether the best 2-sensor
layout plateaus while 4-sensor layouts keep improving.

Call :meth:`record` once per generation, right where the logbook is recorded.
Generations where a given count is absent (it went extinct, or never appeared)
leave a gap in that count's series via :meth:`series`, so plotted lines break
rather than interpolating across the gap.
"""

from __future__ import annotations

import math

from config.params import Population


class PerLengthEvolution:
    """Per-generation, per-sensor-count fitness statistics."""

    def __init__(self) -> None:
        self._records: list[dict] = []

    def record(self, gen: int, population: Population) -> None:
        """Snapshot one generation, grouping valid fitnesses by sensor count."""
        groups: dict[int, list[float]] = {}
        for ind in population:
            if not ind.fitness.valid:
                continue
            groups.setdefault(len(ind), []).append(ind.fitness.values[0])

        per_length = {
            length: {
                "max": max(vals),
                "avg": sum(vals) / len(vals),
                "min": min(vals),
                "count": len(vals),
            }
            for length, vals in groups.items()
        }
        self._records.append({"gen": gen, "per_length": per_length})

    @property
    def generations(self) -> list[int]:
        """Recorded generation indices, in order."""
        return [r["gen"] for r in self._records]

    @property
    def lengths(self) -> list[int]:
        """Every sensor count observed across all recorded generations."""
        seen: set[int] = set()
        for r in self._records:
            seen.update(r["per_length"])
        return sorted(seen)

    def series(self, length: int, metric: str = "max") -> tuple[list[int], list[float]]:
        """Return ``(generations, values)`` for one sensor count.

        ``values`` is ``NaN`` in any generation where that count had no valid
        individuals, so the plotted line breaks across extinction gaps.

        Args:
            length: the sensor count to extract.
            metric: one of ``"max"``, ``"avg"``, ``"min"``, ``"count"``.
        """
        gens: list[int] = []
        vals: list[float] = []
        for r in self._records:
            gens.append(r["gen"])
            stat = r["per_length"].get(length)
            vals.append(float(stat[metric]) if stat is not None else math.nan)
        return gens, vals

    def __len__(self) -> int:
        return len(self._records)
