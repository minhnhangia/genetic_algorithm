from dataclasses import dataclass
from enum import Enum

class SensorType(Enum):
    LIDAR_16_CH = 1     # e.g., 16-channel spinning LiDAR
    LIDAR_32_CH = 2     # e.g., 32-channel spinning LiDAR
    SOLID_STATE = 3     # e.g., Directional solid-state LiDAR


@dataclass(frozen=True)
class Sensor:
    sensor_type: SensorType
    price: float
    fov_horizontal_deg: float
    fov_vertical_deg: float
    range_m: float


DEFAULT_SENSORS = [
    Sensor(
        sensor_type=SensorType.LIDAR_16_CH,
        price=399.0,
        fov_horizontal_deg=360.0,
        fov_vertical_deg=30.0,
        range_m=100.0,
    ),
    Sensor(
        sensor_type=SensorType.LIDAR_32_CH,
        price=799.0,
        fov_horizontal_deg=360.0,
        fov_vertical_deg=40.0,
        range_m=120.0,
    ),
    Sensor(
        sensor_type=SensorType.SOLID_STATE,
        price=1299.0,
        fov_horizontal_deg=120.0,
        fov_vertical_deg=45.0,
        range_m=80.0,
    ),
]