import trimesh
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import open3d as o3d
import os

# --- 1. FILE PATHS ---
# Adjust this base path to point to your local robotnik_description folder
BASE_PKG_DIR = os.path.expanduser(
    "~/genetic_algorithm_ws/ros2_ws/src/robotnik_description"
)
CHASSIS_MESH_PATH = os.path.join(
    BASE_PKG_DIR, "meshes/bases/rbkairos/rbkairos_chassis.stl"
)


def build_robot_surface() -> trimesh.Trimesh:
    print("Loading meshes...")
    chassis: trimesh.Trimesh = trimesh.load_mesh(CHASSIS_MESH_PATH)

    combined_mesh: trimesh.Trimesh = chassis
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
    # We offset the ray origin 5mm along the normal to prevent the ray from
    # immediately colliding with the face it is spawned from.
    ray_origins = pts_up + (nrm_up * 0.005)
    ray_directions = nrm_up

    # intersects_any returns True if the ray hits the mesh, False if it escapes to infinity
    hits = mesh.ray.intersects_any(ray_origins, ray_directions)

    # We only want to keep points that DO NOT hit anything (~ inverts the boolean array)
    exterior_mask = ~hits

    valid_points = pts_up[exterior_mask]
    valid_normals = nrm_up[exterior_mask]

    # OFFSET: Move points 5cm (0.05 meters) outward from the true exterior surface
    valid_points = valid_points + (valid_normals * 0.05)

    filtered_points, filtered_normals = interactive_crop_points(
        valid_points, valid_normals
    )

    print(f"Filtered down to {len(filtered_points)} valid mounting nodes.")

    # --- 5. BUILD THE MUTATION GRAPH ---
    print(f"Constructing KDTree with a neighbor radius of {neighbor_radius}m...")
    tree = cKDTree(filtered_points)

    # Create a NetworkX graph to represent valid mutations (slides)
    mutation_graph = nx.Graph()

    for i, point in enumerate(filtered_points):
        # Store the (x,y,z) coordinate and normal vector as node attributes
        mutation_graph.add_node(i, pos=point, normal=filtered_normals[i])

        # Find all valid neighbor nodes within the specified radius
        neighbors = tree.query_ball_point(point, r=neighbor_radius)
        for neighbor_idx in neighbors:
            if neighbor_idx != i:
                mutation_graph.add_edge(i, neighbor_idx)

    print(
        f"Graph constructed: {mutation_graph.number_of_nodes()} nodes, {mutation_graph.number_of_edges()} edges."
    )
    return mutation_graph, filtered_points


def interactive_crop_points(valid_points, valid_normals):
    print("Opening Open3D Visualizer...")
    print("INSTRUCTIONS:")
    print("1. Press 'K' to lock the view and enter Selection Mode.")
    print(
        "2. Hold 'Ctrl' + Left Click to draw a polygon around the points you want to KEEP."
    )
    print("3. Press 'C' to crop (delete everything outside the polygon).")
    print("4. Press 'S' to finalize the selection, then close the window.")

    if len(valid_points) == 0:
        print("No points available for cropping. Returning original points.")
        return valid_points, valid_normals

    # Convert your numpy points to an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.normals = o3d.utility.Vector3dVector(valid_normals)

    # Color them red to match your previous visualizer
    pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(valid_points), 1)))

    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(window_name="Open3D Crop Editor")
    visualizer.add_geometry(pcd)
    visualizer.run()
    cropped_geometry = visualizer.get_cropped_geometry()
    visualizer.destroy_window()

    if cropped_geometry is None:
        print("No crop was saved. Returning original points.")
        return valid_points, valid_normals

    curated_points = np.asarray(cropped_geometry.points)
    if curated_points.size == 0:
        print(
            "The crop was empty. Returning original points instead of producing an empty graph."
        )
        return valid_points, valid_normals

    if not cropped_geometry.has_normals():
        print(
            "The cropped geometry did not preserve normals. Returning original points."
        )
        return valid_points, valid_normals

    curated_normals = np.asarray(cropped_geometry.normals)
    if curated_normals.shape != curated_points.shape:
        print(
            "The cropped geometry normals do not match the selected points. Returning original points."
        )
        return valid_points, valid_normals

    print(
        f"Curation complete. Reduced from {len(valid_points)} to {len(curated_points)} nodes."
    )
    return curated_points, curated_normals


def visualize_ga_graph(
    mesh: trimesh.Trimesh, graph: nx.Graph, node_positions: np.ndarray
) -> None:
    print("Visualizing the node graph. Close the window to exit.")

    if len(node_positions) == 0:
        print("No mounting nodes remain after cropping. Showing the mesh only.")
        mesh.visual.face_colors = [100, 100, 100, 100]
        trimesh.Scene([mesh]).show()
        return

    # Create a point cloud for the nodes.
    node_cloud = trimesh.points.PointCloud(node_positions, colors=[255, 0, 0, 255])

    # Convert graph edges into line segments so the connectivity is visible.
    edge_segments = np.array(
        [[graph.nodes[u]["pos"], graph.nodes[v]["pos"]] for u, v in graph.edges()]
    )
    edge_path = trimesh.load_path(edge_segments) if len(edge_segments) else None
    if edge_path is not None:
        edge_path.colors = np.tile([180, 180, 180, 60], (len(edge_path.entities), 1))

    mesh.visual.face_colors = [100, 100, 100, 100]
    scene_items = [mesh, node_cloud]
    if edge_path is not None:
        scene_items.append(edge_path)

    scene = trimesh.Scene(scene_items)
    scene.show(line_settings={"line_width": 1})


if __name__ == "__main__":
    robot_mesh = build_robot_surface()

    # Parameters for your GA:
    # point_count: How fine-grained your optimization space is.
    # neighbor_radius: The maximum distance a sensor can "slide" during a mutation.
    ga_graph, node_positions = generate_ga_graph(
        robot_mesh, point_count=50000, neighbor_radius=0.05
    )

    visualize_ga_graph(robot_mesh, ga_graph, node_positions)
