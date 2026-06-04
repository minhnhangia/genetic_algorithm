from config.sensors import Sensor

import numpy as np

def generate_sensor_rays(sensor: Sensor) -> np.ndarray:
    """
    Generates unit direction vectors for a given sensor's specifications.
    """
    # 1. Calculate the number of steps based on FOV and resolution
    num_horizontal_steps = int(sensor.fov_horizontal_deg / sensor.horizontal_res_deg)
    num_vertical_steps = sensor.vertical_channels

    # 2. Generate linearly spaced angles (in radians)
    # Center the FOV around 0 (straight ahead / horizontal)
    h_angles = np.deg2rad(np.linspace(
        -sensor.fov_horizontal_deg / 2, 
        sensor.fov_horizontal_deg / 2, 
        num_horizontal_steps,
        endpoint=False # Avoid overlapping the 360th degree with the 0th degree
    ))
    
    # If 1 channel (Solid State), point straight ahead (0 degrees)
    if num_vertical_steps == 1:
        v_angles = np.array([0.0])
    else:
        v_angles = np.deg2rad(np.linspace(
            -sensor.fov_vertical_deg / 2, 
            sensor.fov_vertical_deg / 2, 
            num_vertical_steps
        ))

    # 3. Create a meshgrid of all angle combinations
    H, V = np.meshgrid(h_angles, v_angles)
    H = H.flatten()
    V = V.flatten()

    # 4. Convert Spherical to Cartesian Coordinates (Unit Vectors)
    # x = forward, y = left, z = up (Standard Robotics ROS convention)
    x = np.cos(V) * np.cos(H)
    y = np.cos(V) * np.sin(H)
    z = np.sin(V)

    ray_directions = np.stack((x, y, z), axis=-1)
    
    # Optional: Apply range limits before returning
    # You would use `sensor.range_m` in Open3D's raycasting scene to limit hit distances.
    return ray_directions

# Example Usage:
# solid_state_rays = generate_sensor_rays(SENSOR_CATALOG[SensorType.SOLID_STATE])
# Returns a shape of (1200, 3) vectors for the Solid State LiDAR