from dataclasses import dataclass

from .sensors import Sensor
from .graph import MOUNTING_GRAPH

Individual = list["Gene"]
Population = list[Individual]


@dataclass
class Gene:
    """
    Represents a single sensor placement in the individual's layout.

    Attributes:
        sensor: The Sensor object representing the type and specs of the sensor.
        node_id: The ID of the node where this sensor is mounted (integer key in MOUNTING_GRAPH).
        pitch: The pitch angle of the sensor in degrees (-90 to +90).
        roll: The roll angle of the sensor in degrees (-90 to +90).
        yaw: The yaw angle of the sensor in degrees (-180 to +180).
    """

    sensor: Sensor
    node_id: int  # integer key of a node in MOUNTING_GRAPH
    pitch: int
    roll: int
    yaw: int


MAX_SENSORS_PER_INDIVIDUAL: int = 4

# Minimum Euclidean separation (meters) kept between sensors, so they don't
# cluster together. Used both when initializing an individual and when the Add
# mutation places a new sensor. Applied via Poisson-disk rejection sampling that
# relaxes the threshold if it can't be met (see utils.select_spread_nodes).
# Larger => more spread.
MIN_SENSOR_SEPARATION_M: float = 0.3

# Minimum horizontal angular resolution (degrees) used when casting a sensor's
# rays for fitness evaluation. A sensor whose native horizontal_res_deg is finer
# than this is coarsened to this floor, dropping rays that oversample the azimuth
# grid (e.g. a solid-state unit's 0.08 deg vs ~1 deg bins) without changing
# coverage; sensors already coarser are untouched. This is the dominant eval-cost
# knob -- raise it to go faster (with some fitness drift), lower it for fidelity.
# Note: the *vertical* axis is deliberately NOT capped -- for a narrow-FOV
# directional sensor it maps to radial ground resolution at grazing angles, where
# fewer rays measurably under-counts coverage. See raycasting.local_rays.
EVAL_MIN_HORIZONTAL_RES_DEG: float = 0.25

VALID_NODE_IDS: list[int] = sorted(MOUNTING_GRAPH.nodes())

POPULATION_SIZE: int = 5000

# 0.5% of the population as elite individuals to carry over unchanged to the next generation
ELITE_COUNT: int = max(1, round(POPULATION_SIZE * 0.005))

NGEN: int = 50
