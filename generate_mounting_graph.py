import trimesh
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import os

# --- 1. FILE PATHS ---
# Adjust this base path to point to your local robotnik_description folder
BASE_PKG_DIR = os.path.expanduser(
    "~/genetic_algorithm_ws/ros2_ws/src/robotnik_description"
)
CHASSIS_MESH_PATH = os.path.join(
    BASE_PKG_DIR, "meshes/bases/rbkairos/rbkairos_chassis.stl"
)
TOP_COVER_MESH_PATH = os.path.join(
    BASE_PKG_DIR, "meshes/bases/rbkairos/rbkairos_top_cover.stl"
)


def build_robot_surface() -> trimesh.Trimesh:
    print("Loading meshes...")
    chassis: trimesh.Trimesh = trimesh.load_mesh(CHASSIS_MESH_PATH)
    top_cover: trimesh.Trimesh = trimesh.load_mesh(TOP_COVER_MESH_PATH)

    # --- 2. APPLY KINEMATIC TRANSFORMS ---
    # The top cover has a joint offset (0.56162) + visual offset (0.0065) = 0.56812
    transformation_matrix = np.eye(4)
    transformation_matrix[2, 3] = 0.56162 + 0.0065
    top_cover.apply_transform(transformation_matrix)

    # --- 3. MERGE GEOMETRY ---
    # Combine them into a single watertight (or near-watertight) scene
    combined_mesh: trimesh.Trimesh = trimesh.util.concatenate([chassis, top_cover])
    print(f"Combined mesh created with {len(combined_mesh.faces)} faces.")
    return combined_mesh


def generate_ga_graph(
    mesh: trimesh.Trimesh, point_count: int = 2000, neighbor_radius: float = 0.10
) -> tuple[nx.Graph, np.ndarray]:
    print(f"Sampling {point_count} points evenly across the robot surface...")

    # --- 4. POISSON-DISK APPROXIMATION ---
    points, face_indices = trimesh.sample.sample_surface_even(
        mesh=mesh, count=point_count
    )

    normals = mesh.face_normals[face_indices]

    # Remove downward-facing points
    non_downward_mask = normals[:, 2] >= -0.5
    pts_up = points[non_downward_mask]
    nrm_up = normals[non_downward_mask]

    # --- Raycast to remove internal geometry ---
    print("Running raycast occlusion check to remove internal points...")
    # We offset the ray origin 1mm along the normal to prevent the ray from
    # immediately colliding with the face it is spawned from.
    ray_origins = pts_up + (nrm_up * 0.001)
    ray_directions = nrm_up

    # intersects_any returns True if the ray hits the mesh, False if it escapes to infinity
    hits = mesh.ray.intersects_any(ray_origins, ray_directions)

    # We only want to keep points that DO NOT hit anything (~ inverts the boolean array)
    exterior_mask = ~hits

    valid_points = pts_up[exterior_mask]
    valid_normals = nrm_up[exterior_mask]

    # OFFSET: Move points 5cm (0.05 meters) outward from the true exterior surface
    valid_points = valid_points + (valid_normals * 0.05)

    print(f"Filtered down to {len(valid_points)} valid mounting nodes.")

    # --- 5. BUILD THE MUTATION GRAPH ---
    print(f"Constructing KDTree with a neighbor radius of {neighbor_radius}m...")
    tree = cKDTree(valid_points)

    # Create a NetworkX graph to represent valid mutations (slides)
    mutation_graph = nx.Graph()

    for i, point in enumerate(valid_points):
        # Store the (x,y,z) coordinate and normal vector as node attributes
        mutation_graph.add_node(i, pos=point, normal=valid_normals[i])

        # Find all valid neighbor nodes within the specified radius
        neighbors = tree.query_ball_point(point, r=neighbor_radius)
        for neighbor_idx in neighbors:
            if neighbor_idx != i:
                mutation_graph.add_edge(i, neighbor_idx)

    print(
        f"Graph constructed: {mutation_graph.number_of_nodes()} nodes, {mutation_graph.number_of_edges()} edges."
    )
    return mutation_graph, valid_points


if __name__ == "__main__":
    robot_mesh = build_robot_surface()

    # Parameters for your GA:
    # point_count: How fine-grained your optimization space is.
    # neighbor_radius: The maximum distance a sensor can "slide" during a mutation.
    ga_graph, node_positions = generate_ga_graph(
        robot_mesh, point_count=20000, neighbor_radius=0.05
    )

    # --- VISUALIZATION (Sanity Check) ---
    print("Visualizing the node graph. Close the window to exit.")

    # Create a point cloud for the nodes
    pc = trimesh.points.PointCloud(node_positions, colors=[255, 0, 0, 255])  # Red nodes

    # Render the transparent robot mesh + the discrete GA mounting nodes
    robot_mesh.visual.face_colors = [100, 100, 100, 100]  # Transparent grey
    scene = trimesh.Scene([robot_mesh, pc])
    scene.show()
