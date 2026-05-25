from .params import Gene
from . import sensors as sensors

# Seeding a Population
# Initializes a population containing expert seeds, filling the rest 
# with random individuals to maintain genetic diversity.
# https://deap.readthedocs.io/en/master/tutorials/basic/part1.html#seeding-a-population


# Add or remove seeded individuals here.
# Each seed is a list of Gene objects and will be wrapped in creator.Individual.
SEED_INDIVIDUALS = [
    [
        Gene(sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_16_CH], node_id=5, pitch=0, roll=0),
    ],
    [
        Gene(sensor=sensors.SENSOR_CATALOG[sensors.SensorType.SOLID_STATE], node_id=42, pitch=10, roll=-15),
        Gene(sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_32_CH], node_id=99, pitch=-5, roll=20),
    ],
]