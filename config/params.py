from dataclasses import dataclass

from .sensors import Sensor

type Individual = list[Gene]
type Population = list[Individual]

@dataclass
class Gene:
    """
    Represents a single sensor placement in the individual's layout.
        
    Attributes:
        sensor: The Sensor object representing the type and specs of the sensor.
        node_id: The ID of the node where this sensor is mounted (0-199).
        pitch: The pitch angle of the sensor in degrees (-90 to +90).
        roll: The roll angle of the sensor in degrees (-90 to +90).
        yaw: The yaw angle of the sensor in degrees (-180 to +180).
    """
    sensor: Sensor
    node_id: int    # TODO: represent node_id as a graph/mesh
    pitch: int
    roll: int
    yaw: int

MAX_SENSORS_PER_INDIVIDUAL : int = 4

VALID_NODE_IDS : list[int] = list(range(0, 200))

POPULATION_SIZE : int = 1000