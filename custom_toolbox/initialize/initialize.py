import random

from config import params, sensors
from config.params import Gene, Population, Individual


def create_gene(assigned_node_id: int) -> Gene:
    """Generates a single active sensor configuration."""
    sensor = random.choice(list(sensors.SENSOR_CATALOG.values()))

    # Define physical rotation bounds.
    # Note: If you are using 360-degree spinning LiDARs, roll/pitch is sufficient.
    # If using directional solid-state LiDARs, you may need to swap 'roll' for 'yaw'.
    pitch = random.randint(-90, 90)
    roll = random.randint(-90, 90)
    yaw = random.randint(-180, 180)

    return Gene(
        sensor=sensor, node_id=assigned_node_id, pitch=pitch, roll=roll, yaw=yaw
    )


def create_individual(icls: type[Individual]) -> Individual:
    """Generates an individual with 1 to 4 active sensors."""
    num_sensors = random.randint(1, params.MAX_SENSORS_PER_INDIVIDUAL)

    # Ensure unique node IDs for each sensor in the individual.
    unique_nodes = random.sample(params.VALID_NODE_IDS, num_sensors)

    genes = [create_gene(node_id) for node_id in unique_nodes]

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
