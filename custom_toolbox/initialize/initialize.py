import random

from config import params, sensors
from config.params import Gene, Population, Individual
from custom_toolbox.initialize.orientation import spawn_orientation
from custom_toolbox.utils.utils import select_spread_nodes


def create_gene(assigned_node_id: int, *, oriented: bool = True) -> Gene:
    """Generates a single active sensor configuration.

    ``oriented`` (default) aims the sensor outward + slightly down to avoid
    self-occlusion; pass False for the fully random baseline.
    """
    sensor = random.choice(list(sensors.SENSOR_CATALOG.values()))

    if oriented:
        pitch, roll, yaw = spawn_orientation(assigned_node_id, sensor)
    else:
        pitch = random.randint(-60, 60)
        roll = random.randint(-60, 60)
        yaw = random.randint(-180, 180)

    return Gene(
        sensor=sensor, node_id=assigned_node_id, pitch=pitch, roll=roll, yaw=yaw
    )


def create_individual(icls: type[Individual], *, oriented: bool = True) -> Individual:
    """Generates an individual with 1 to 4 active sensors."""
    num_sensors = random.randint(1, params.MAX_SENSORS_PER_INDIVIDUAL)

    # Unique node IDs, spread out so sensors don't start clustered together.
    unique_nodes = select_spread_nodes(num_sensors, params.MIN_SENSOR_SEPARATION_M)

    genes = [create_gene(node_id, oriented=oriented) for node_id in unique_nodes]

    # Initialize the DEAP list class with our generated genes.
    return icls(genes)


def create_seeded_population(
    icls: type[Individual],
    individual_creator: callable,
    seed_contents: list[Individual],
    population_size: int,
    shuffle=False,
) -> Population:
    """Builds a population from seeded individuals plus random fill."""
    seeded_population = [icls(seed) for seed in seed_contents]

    if len(seeded_population) > population_size:
        raise ValueError(
            f"seed_contents contains {len(seeded_population)} individuals, "
            f"but population_size is only {population_size}."
        )

    random_population = [
        individual_creator() for _ in range(population_size - len(seeded_population))
    ]
    population = seeded_population + random_population

    if shuffle:
        random.shuffle(population)

    return population
