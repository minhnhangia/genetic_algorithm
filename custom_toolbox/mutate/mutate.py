import random
from enum import Enum

from config.params import Gene, Individual, Population
from config.params import MAX_SENSORS_PER_INDIVIDUAL, MIN_SENSOR_SEPARATION_M
from config.graph import MOUNTING_GRAPH
from config.sensors import SENSOR_CATALOG
from custom_toolbox.initialize import initialize
from custom_toolbox.mutate.tiers import Tier, TIER_CONFIGS
from custom_toolbox.utils.utils import select_spread_nodes


class AttributeMutationType(Enum):
    ANGLES = "angles"
    POSITION = "position"
    HARDWARE = "hardware"


def _slide_to_neighbor(node_id: int, occupied: set[int], hops: int) -> int:
    """Walk up to ``hops`` graph hops through unoccupied neighbors; stop early if
    a node has no free neighbor. Returns the final node id."""
    for _ in range(hops):
        candidates = [n for n in MOUNTING_GRAPH.neighbors(node_id) if n not in occupied]
        if not candidates:
            break
        occupied.discard(node_id)
        node_id = random.choice(candidates)
        occupied.add(node_id)
    return node_id


def mutate_sensor_layout(individual: Individual) -> tuple[Individual]:
    """Tier-aware mutation. Perturbation intensity (structural/attr probabilities,
    angle sigma, position hops) is read from the individual's ``mutation_tier``;
    defaults to MEDIUM (the original constants) when untagged.

    - Structural: add a sensor at a spread free node, or drop one.
    - ANGLES: Gaussian jitter on pitch/roll/yaw within mechanical limits.
    - POSITION: slide a sensor up to ``position_hops`` unoccupied graph neighbors.
    - HARDWARE: swap the sensor model.
    """
    cfg = TIER_CONFIGS[getattr(individual, "mutation_tier", Tier.MEDIUM)]

    # 1. Structural Mutation (Add / Drop)
    if random.random() < cfg.structural_prob:
        # 50/50 split between Add and Drop
        if random.random() < 0.5 and len(individual) < MAX_SENSORS_PER_INDIVIDUAL:
            # Place the new sensor on a free node spread clear of the existing ones
            new_nodes = select_spread_nodes(
                1,
                MIN_SENSOR_SEPARATION_M,
                existing_nodes=[gene.node_id for gene in individual],
            )
            if new_nodes:
                individual.append(initialize.create_gene(new_nodes[0]))

        elif len(individual) > 1:
            idx = random.randrange(len(individual))
            individual.pop(idx)

    # 2. Attribute Mutation (Jitter / Move / Hardware), per gene
    sigma = cfg.angle_sigma
    for gene in individual:
        if random.random() < cfg.attr_prob:

            mutation_type = random.choice(list(AttributeMutationType))

            if mutation_type == AttributeMutationType.ANGLES:
                gene.pitch = int(
                    round(max(-90, min(90, gene.pitch + random.gauss(0, sigma))))
                )
                gene.roll = int(
                    round(max(-90, min(90, gene.roll + random.gauss(0, sigma))))
                )
                gene.yaw = int(
                    round(max(-180, min(180, gene.yaw + random.gauss(0, sigma))))
                )

            elif mutation_type == AttributeMutationType.POSITION:
                # Slide up to cfg.position_hops unoccupied graph neighbors.
                occupied = set(g.node_id for g in individual)
                gene.node_id = _slide_to_neighbor(
                    gene.node_id, occupied, cfg.position_hops
                )

            elif mutation_type == AttributeMutationType.HARDWARE:
                # Swap the sensor out for a different model from the catalog
                gene.sensor = random.choice(list(SENSOR_CATALOG.values()))

    return (individual,)
