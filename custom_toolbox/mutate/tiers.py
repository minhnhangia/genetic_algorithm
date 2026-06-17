"""Fitness-tiered mutation: split the population into superior/medium/inferior
sub-populations and perturb each with a different intensity.

Superior = exploitation (tiny perturbation); inferior = exploration (large);
medium = balance. MEDIUM config matches the operator's original constants, so
running without tier assignment reproduces the un-tiered behaviour.
"""

from dataclasses import dataclass
from enum import Enum

from config.params import Individual, Population


class Tier(Enum):
    SUPERIOR = "superior"
    MEDIUM = "medium"
    INFERIOR = "inferior"


@dataclass(frozen=True)
class TierConfig:
    structural_prob: float  # P(add/drop a sensor)
    attr_prob: float  # P(mutate each gene)
    angle_sigma: float  # Gaussian sigma (deg) for angle jitter
    position_hops: int  # graph hops a POSITION move may travel


# MEDIUM == the operator's original constants (baseline-preserving).
TIER_CONFIGS: dict[Tier, TierConfig] = {
    Tier.SUPERIOR: TierConfig(
        structural_prob=0.02, attr_prob=0.30, angle_sigma=2.0, position_hops=1
    ),
    Tier.MEDIUM: TierConfig(
        structural_prob=0.10, attr_prob=0.50, angle_sigma=5.0, position_hops=1
    ),
    Tier.INFERIOR: TierConfig(
        structural_prob=0.25, attr_prob=0.70, angle_sigma=15.0, position_hops=2
    ),
}

TIER_PROPORTIONS: dict[Tier, float] = {
    Tier.SUPERIOR: 0.1,
    Tier.MEDIUM: 0.4,
    Tier.INFERIOR: 0.5,
}


def assign_fitness_tiers(
    population: Population,
    proportions: dict[Tier, float] = TIER_PROPORTIONS,
) -> None:
    """Tag each individual with ``mutation_tier`` by global fitness rank.

    Recomputed every call; remainder goes to INFERIOR so counts sum to len(pop).
    Assumes valid fitness (true at the start of a generation step).
    """
    ranked = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
    n = len(ranked)
    n_sup = round(n * proportions[Tier.SUPERIOR])
    n_med = round(n * proportions[Tier.MEDIUM])
    for i, ind in enumerate(ranked):
        if i < n_sup:
            ind.mutation_tier = Tier.SUPERIOR
        elif i < n_sup + n_med:
            ind.mutation_tier = Tier.MEDIUM
        else:
            ind.mutation_tier = Tier.INFERIOR
