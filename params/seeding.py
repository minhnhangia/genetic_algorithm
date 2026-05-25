from params.params import Gene
from params.sensors import DEFAULT_SENSORS

# Seeding a Population
# Initializes a population containing expert seeds, filling the rest 
# with random individuals to maintain genetic diversity.
# https://deap.readthedocs.io/en/master/tutorials/basic/part1.html#seeding-a-population


# Add or remove seeded individuals here.
# Each seed is a list of Gene objects and will be wrapped in creator.Individual.
SEED_INDIVIDUALS = [
    [
        Gene(sensor=DEFAULT_SENSORS[0], node_id=5, pitch=0, roll=0),
    ],
    [
        Gene(sensor=DEFAULT_SENSORS[2], node_id=42, pitch=10, roll=-15),
        Gene(sensor=DEFAULT_SENSORS[1], node_id=99, pitch=-5, roll=20),
    ],
]