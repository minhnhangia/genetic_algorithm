"""Self-occlusion-aware spawn orientation.

Aims a freshly spawned sensor outward (away from the robot body) and slightly
down, so it lands in the scored region from generation 0 instead of pointing into
the chassis or straight up. Boresight maps to world dir
``(cos yaw*cos pitch, sin yaw*cos pitch, sin pitch)``, so azimuth==yaw and
elevation==pitch (see ``sensor_body.rotation_matrix``).
"""

import math
import random

from config.graph import MOUNTING_GRAPH
from config.sensors import Sensor

SPAWN_DOWNTILT_DEG = 25.0      # base downward tilt for directional sensors
SPAWN_JITTER_DEG = 30.0        # +/- exploration jitter around computed angles
VERTICAL_NORMAL_EPS = 0.2      # |horizontal normal| below this -> treat as vertical
SPAWN_360_PITCH_RANGE = (-20.0, 5.0)   # near-upright tilt for 360-deg sensors
SPAWN_360_ROLL_RANGE = (-15.0, 15.0)


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def spawn_orientation(node_id: int, sensor: Sensor) -> tuple[int, int, int]:
    """(pitch, roll, yaw) in degrees for a sensor mounted at ``node_id``."""
    node = MOUNTING_GRAPH.nodes[node_id]
    nx, ny = float(node["normal"][0]), float(node["normal"][1])
    px, py = float(node["pos"][0]), float(node["pos"][1])

    # 360-deg sensors scan a full circle: heading is meaningless, keep upright.
    if sensor.fov_horizontal_deg >= 360.0:
        pitch = random.uniform(*SPAWN_360_PITCH_RANGE)
        roll = random.uniform(*SPAWN_360_ROLL_RANGE)
        yaw = random.uniform(-180.0, 180.0)
        return int(round(pitch)), int(round(roll)), int(round(yaw))

    # Directional sensor: aim outward (normal heading, radial fallback) + downtilt.
    if math.hypot(nx, ny) > VERTICAL_NORMAL_EPS:
        heading = math.atan2(ny, nx)
    elif math.hypot(px, py) > VERTICAL_NORMAL_EPS:
        heading = math.atan2(py, px)
    else:
        heading = math.radians(random.uniform(-180.0, 180.0))

    yaw = _wrap180(math.degrees(heading) + random.uniform(-SPAWN_JITTER_DEG, SPAWN_JITTER_DEG))
    pitch = -SPAWN_DOWNTILT_DEG + random.uniform(-SPAWN_JITTER_DEG, SPAWN_JITTER_DEG)
    pitch = max(-80.0, min(0.0, pitch))  # horizontal-to-down, never up
    roll = random.uniform(-SPAWN_JITTER_DEG, SPAWN_JITTER_DEG)
    return int(round(pitch)), int(round(roll)), int(round(yaw))
