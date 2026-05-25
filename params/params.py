from dataclasses import dataclass
from enum import Enum

class SensorType(Enum):
    LIDAR_16_CH = 1     # e.g., 16-channel spinning LiDAR
    LIDAR_32_CH = 2     # e.g., 32-channel spinning LiDAR
    SOLID_STATE = 3     # e.g., Directional solid-state LiDAR

@dataclass
class Gene:
    sensor_type: SensorType
    node_id: int
    pitch: int
    roll: int

MAX_SENSORS_PER_INDIVIDUAL : int = 4

VALID_NODE_IDS : list[int] = list(range(0, 200))

POPULATION_SIZE : int = 100