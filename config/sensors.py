from dataclasses import dataclass
from enum import Enum


class SensorType(Enum):
    LIDAR_16_CH = 1  # e.g., 16-channel spinning LiDAR
    LIDAR_32_CH = 2  # e.g., 32-channel spinning LiDAR
    SOLID_STATE = 3  # e.g., Directional solid-state LiDAR


@dataclass(frozen=True)
class Sensor:
    """
    Represents a sensor type with its specifications.

    Attributes:
        sensor_type: The type of the sensor.
        price: The price of the sensor.
        fov_horizontal_deg: The horizontal field of view in degrees.
        fov_vertical_deg: The vertical field of view in degrees.
        range_m: The detection range in meters.
        vertical_channels: The number of vertical channels (for LiDARs).
        horizontal_res_deg: The horizontal resolution in degrees (for LiDARs).
        body_radius_m: Radius of the physical sensor body (a short cylinder).
            Used both to render the sensor and to occlude other sensors' rays.
        body_height_m: Height of the physical sensor body cylinder.
    """

    sensor_type: SensorType
    price: float
    fov_horizontal_deg: float
    fov_vertical_deg: float
    range_m: float
    vertical_channels: int
    horizontal_res_deg: float
    body_radius_m: float = 0.05
    body_height_m: float = 0.07


SENSOR_CATALOG = {
    SensorType.LIDAR_16_CH: Sensor(
        sensor_type=SensorType.LIDAR_16_CH,
        price=399.0,
        fov_horizontal_deg=360.0,
        fov_vertical_deg=30.0,
        range_m=100.0,
        vertical_channels=16,
        horizontal_res_deg=0.2,
        body_radius_m=0.052,  # ~103 mm diameter (VLP-16 class)
        body_height_m=0.072,
    ),
    SensorType.LIDAR_32_CH: Sensor(
        sensor_type=SensorType.LIDAR_32_CH,
        price=799.0,
        fov_horizontal_deg=360.0,
        fov_vertical_deg=40.0,
        range_m=120.0,
        vertical_channels=32,
        horizontal_res_deg=0.1,
        body_radius_m=0.058,  # ~116 mm diameter (HDL-32 class)
        body_height_m=0.090,
    ),
    SensorType.SOLID_STATE: Sensor(
        sensor_type=SensorType.SOLID_STATE,
        price=299.0,
        fov_horizontal_deg=120.0,
        fov_vertical_deg=45.0,
        range_m=80.0,
        vertical_channels=1,
        horizontal_res_deg=0.1,
        body_radius_m=0.040,  # compact solid-state unit (cylinder proxy)
        body_height_m=0.040,
    ),
}
