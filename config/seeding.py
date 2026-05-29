from .params import Gene, VALID_NODE_IDS
from . import sensors as sensors

# Seeding a Population
# Initializes a population containing expert seeds, filling the rest
# with random individuals to maintain genetic diversity.
# https://deap.readthedocs.io/en/master/tutorials/basic/part1.html#seeding-a-population

# Node IDs are drawn from the real mounting graph via VALID_NODE_IDS.
# The first three sorted node IDs are used as safe placeholders.
# Replace them with expert-chosen positions once you have explored the
# generated graph visually (e.g. via visualize_ga_graph).
_n = VALID_NODE_IDS

# Add or remove seeded individuals here.
# Each seed is a list of Gene objects and will be wrapped in creator.Individual.
SEED_INDIVIDUALS = [
    [
        Gene(sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_16_CH], node_id=_n[0], pitch=0, roll=0, yaw=0),
    ],
    [
        Gene(sensor=sensors.SENSOR_CATALOG[sensors.SensorType.SOLID_STATE], node_id=_n[1], pitch=10, roll=-15, yaw=0),
        Gene(sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_32_CH], node_id=_n[2], pitch=-5, roll=20, yaw=0),
    ],
]