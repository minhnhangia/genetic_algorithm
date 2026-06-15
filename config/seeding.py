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
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_32_CH],
            node_id=1642,
            pitch=-10,
            roll=-20,
            yaw=167,
        ),
    ],
    [
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_32_CH],
            node_id=1318,
            pitch=26,
            roll=-10,
            yaw=-121,
        ),
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_16_CH],
            node_id=1642,
            pitch=-19,
            roll=-18,
            yaw=96,
        ),
    ],
    [
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_16_CH],
            node_id=1642,
            pitch=26,
            roll=12,
            yaw=8,
        ),
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_32_CH],
            node_id=3765,
            pitch=-9,
            roll=-31,
            yaw=-125,
        ),
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.SOLID_STATE],
            node_id=5022,
            pitch=+28,
            roll=-6,
            yaw=-122,
        ),
    ],
    [
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.SOLID_STATE],
            node_id=1642,
            pitch=24,
            roll=17,
            yaw=2,
        ),
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_32_CH],
            node_id=3765,
            pitch=-9,
            roll=-31,
            yaw=-125,
        ),
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.SOLID_STATE],
            node_id=5022,
            pitch=+27,
            roll=-11,
            yaw=-118,
        ),
        Gene(
            sensor=sensors.SENSOR_CATALOG[sensors.SensorType.LIDAR_16_CH],
            node_id=39,
            pitch=15,
            roll=11,
            yaw=-127,
        ),
    ],
]
