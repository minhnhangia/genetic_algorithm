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
    node_id: int    # integer key of a node in MOUNTING_GRAPH
    pitch: int
    roll: int
    yaw: int

MAX_SENSORS_PER_INDIVIDUAL : int = 4

VALID_NODE_IDS: list[int] = sorted(MOUNTING_GRAPH.nodes())

POPULATION_SIZE : int = 1000

ELITE_COUNT : int = max(1, round(POPULATION_SIZE * 0.005))  # 0.5% of the population as elite individuals to carry over unchanged to the next generation

NGEN : int = 100