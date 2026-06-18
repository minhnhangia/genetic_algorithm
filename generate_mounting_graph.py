import pickle
import pathlib
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import open3d as o3d
import os

GRAPH_SAVE_PATH = pathlib.Path(__file__).parent / "data" / "mounting_graph.pkl"


def save_graph(graph: nx.Graph, path: pathlib.Path = GRAPH_SAVE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"Graph saved to {path}  ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )


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
    mesh: trimesh.Trimesh,
    point_count: int = 2000,
    neighbor_radius: float = 0.10,
    curate: bool = True,
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
    valid_points = valid_points + (valid_normals * 0.01)

    if curate:
        filtered_points, filtered_normals = interactive_crop_points(
            valid_points, valid_normals
        )
    else:
        # Batch / non-interactive: rely on the automated downward + internal-raycast
        # filters above and skip the manual Open3D crop.
        filtered_points, filtered_normals = valid_points, valid_normals

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
    print("INSTRUCTIONS:")
    print("  1. Press 'K' to lock the view and enter Selection Mode.")
    print("  2. Hold 'Ctrl' + Left Click to draw a polygon around points to REMOVE.")
    print("  3. Press 'C' to confirm — NOTE: after 'C' only the points being")
    print("     REMOVED are shown. Press 'S' to apply and open the next round.")
    print("  4. Close the window WITHOUT pressing 'S' when you are done.")

    if len(valid_points) == 0:
        print("No points available. Returning original points.")
        return valid_points, valid_normals

    current_points = valid_points
    current_normals = valid_normals
    round_num = 1

    while True:
        print(
            f"\n--- Removal round {round_num} ({len(current_points)} points remaining) ---"
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(current_points)
        pcd.normals = o3d.utility.Vector3dVector(current_normals)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile([1, 0, 0], (len(current_points), 1))
        )

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(
            window_name=f"Round {round_num}: select points to REMOVE — close window to finish"
        )
        vis.add_geometry(pcd)
        vis.run()
        selected = vis.get_cropped_geometry()
        vis.destroy_window()

        # None means the user closed without pressing S → curation is done.
        if selected is None:
            print("Window closed. Finishing curation.")
            break

        if not isinstance(selected, o3d.geometry.PointCloud):
            print("Unexpected selection type. Finishing curation.")
            break

        # get_cropped_geometry() returns the points INSIDE the polygon, which
        # are the ones to remove. We keep everything NOT matched by a KDTree
        # lookup. The 1 µm threshold absorbs any float rounding in O3D's C++ layer.
        removed_pts = np.asarray(selected.points)
        if removed_pts.size == 0:
            print("No points were selected. Finishing curation.")
            break

        remove_tree = cKDTree(removed_pts)
        distances, _ = remove_tree.query(current_points)
        kept_mask = distances > 1e-6

        if not kept_mask.any():
            print(
                "All remaining points were selected. Reverting this round and finishing."
            )
            break

        removed_count = int((~kept_mask).sum())
        current_points = current_points[kept_mask]
        current_normals = current_normals[kept_mask]

        print(f"Removed {removed_count} points. {len(current_points)} remaining.")
        round_num += 1

    total_removed = len(valid_points) - len(current_points)
    print(
        f"\nCuration complete. Removed {total_removed} points total. "
        f"Kept {len(current_points)} of {len(valid_points)} nodes."
    )
    return current_points, current_normals


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
    save_graph(ga_graph)
