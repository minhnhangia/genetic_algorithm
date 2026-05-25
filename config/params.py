from dataclasses import dataclass

from .sensors import Sensor

@dataclass
class Gene:
    sensor: Sensor
    node_id: int
    pitch: int
    roll: int

MAX_SENSORS_PER_INDIVIDUAL : int = 4

VALID_NODE_IDS : list[int] = list(range(0, 200))

POPULATION_SIZE : int = 100