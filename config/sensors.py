from dataclasses import dataclass
from enum import Enum


class SensorType(Enum):
    LIDAR_16_CH = 1  # e.g., Hesai XT16
    LIDAR_32_CH = 2  # e.g., Hesai XT32
    SOLID_STATE = 3  # e.g., Hesai ATX


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
    # Hesai XT16
    SensorType.LIDAR_16_CH: Sensor(
        sensor_type=SensorType.LIDAR_16_CH,
        price=2600.0,
        fov_horizontal_deg=360.0,
        fov_vertical_deg=30.0,  # Covers ±15°
        range_m=120.0,
        vertical_channels=16,
        horizontal_res_deg=0.18,  # 10Hz
        body_radius_m=0.038,
        body_height_m=0.1032,
    ),
    # Hesai XT32
    SensorType.LIDAR_32_CH: Sensor(
        sensor_type=SensorType.LIDAR_32_CH,
        price=3520.0,
        fov_horizontal_deg=360.0,
        fov_vertical_deg=31.0,  # Covers -16° to +15°
        range_m=120.0,
        vertical_channels=32,
        horizontal_res_deg=0.18,  # 10Hz
        body_radius_m=0.038,
        body_height_m=0.1032,
    ),
    # Hesai ATX
    SensorType.SOLID_STATE: Sensor(
        sensor_type=SensorType.SOLID_STATE,
        price=1980.0,
        fov_horizontal_deg=120.0,
        fov_vertical_deg=20.0,
        range_m=230.0,
        vertical_channels=256,
        horizontal_res_deg=0.08,
        body_radius_m=0.050,S
        body_height_m=0.030,
    ),
}
