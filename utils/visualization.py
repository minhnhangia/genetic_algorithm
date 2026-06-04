from IPython.display import HTML, display

from config.params import Gene, Population, Individual
from config.sensors import SensorType


def visualize_population(population: Population, max_display: int = 5) -> None:
    sensor_colors = {
        1: "#1f77b4",
        2: "#ff7f0e",
        3: "#2ca02c",
    }

    visible_population = population[:max_display]
    max_genes = max(len(individual) for individual in visible_population)
    rows = []

    for index, individual in enumerate(visible_population, start=1):
        cells = [f"<th>Individual {index}</th>"]

        for gene in individual:
            color = sensor_colors[gene.sensor.sensor_type.value]
            cells.append(
                f'<td style="border-left: 6px solid {color}; background: {color}22;">'
                f'<div style="font-weight: 700; margin-bottom: 4px;">{gene.sensor.sensor_type.name}</div>'
                # f'<div>price ${gene.sensor.price:,.2f}</div>'
                # f'<div>FOV H {gene.sensor.fov_horizontal_deg}°, FOV V {gene.sensor.fov_vertical_deg}°</div>'
                # f'<div>range {gene.sensor.range_m} m</div>'
                f"<div>node {gene.node_id}</div>"
                f"<div>pitch {gene.pitch}, roll {gene.roll}, yaw {gene.yaw}</div>"
                f"</td>"
            )

        while len(cells) < max_genes + 1:
            cells.append('<td class="empty">&mdash;</td>')

        rows.append("<tr>" + "".join(cells) + "</tr>")

    header = (
        "<tr><th>Individual</th>"
        + "".join(f"<th>Sensor {i}</th>" for i in range(1, max_genes + 1))
        + "</tr>"
    )
    summary = ""

    if len(population) > max_display:
        summary = (
            f'<div style="margin: 0 0 12px 0; color: #57606a; font-size: 13px;">'
            f"Showing first {max_display} of {len(population)} individuals."
            f"</div>"
        )

    html = f"""
    <style>
      .population-table {{
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 14px;
      }}
      .population-table th, .population-table td {{
        border: 1px solid #d0d7de;
        padding: 10px 12px;
        vertical-align: top;
      }}
      .population-table th {{
        background: #f6f8fa;
        text-align: center;
      }}
      .population-table .empty {{
        color: #8c959f;
        text-align: center;
      }}
    </style>
    {summary}
    <table class='population-table'>
      {header}
      {''.join(rows)}
    </table>
    """

    display(HTML(html))


def visualize_best_layout(
    individual: Individual,
    evaluator=None,
    *,
    show_rays: bool = True,
    show_ground: bool = True,
    show_cyl: bool = True,
    show_arrows: bool = False,
    max_rays_per_sensor: int = 200,
    include_misses: bool = False,
) -> None:
    """Render the best sensor layout with its ray-cast coverage in a 3D viewer.

    Overlaid on the (semi-transparent) robot mesh:
      * a colored sphere at each mounting node (sensor-type legend), optionally
        with a pointing arrow,
      * the rays cast from every sensor, colored by what they strike:
        red = blocked by chassis (self-occlusion), green = ground plane,
        blue = cylinder wall, grey = miss (only if ``include_misses``),
      * the coverage the fitness actually counts: S_gnd as green points on the
        ground plane and S_cyl as blue points on the radius-R_max wall.

    Args:
        evaluator: a ``CoverageEvaluator`` to reuse. If ``None``, one is built
            (which reloads the chassis mesh and raycasting scene).
        show_rays / show_ground / show_cyl / show_arrows: toggle overlays.
        max_rays_per_sensor: cap on rays drawn per sensor (kept legible).
        include_misses: also draw rays that strike nothing within range.
    """
    import numpy as np
    import trimesh
    from trimesh.path import Path3D
    from trimesh.path.entities import Line
    from IPython.display import display

    from config.graph import MOUNTING_GRAPH
    from generate_mounting_graph import build_robot_surface
    from custom_toolbox.evaluate.evaluate_fitness_raycast import CoverageEvaluator

    # Sensor type → RGBA (matches the HTML table colors)
    SENSOR_COLORS: dict[SensorType, list[int]] = {
        SensorType.LIDAR_16_CH: [31, 119, 180, 230],
        SensorType.LIDAR_32_CH: [255, 127, 14, 230],
        SensorType.SOLID_STATE: [44, 160, 44, 230],
    }

    # Ray strike category → RGBA (alpha conveys importance; misses fade out)
    RAY_COLORS: dict[int, list[int]] = {
        CoverageEvaluator.RAY_BLOCKED: [220, 60, 60, 90],
        CoverageEvaluator.RAY_GROUND: [60, 200, 90, 140],
        CoverageEvaluator.RAY_CYLINDER: [70, 140, 235, 140],
        CoverageEvaluator.RAY_MISS: [160, 160, 160, 35],
    }

    if evaluator is None:
        evaluator = CoverageEvaluator()

    debug = evaluator.coverage_debug(
        individual,
        max_rays_per_sensor=max_rays_per_sensor,
        include_misses=include_misses,
    )

    print(
        f"Best layout  ({len(individual)} sensor{'s' if len(individual) != 1 else ''}, "
        f"fitness={debug['fitness']:.4f}):"
    )
    for i, gene in enumerate(individual, 1):
        pos = MOUNTING_GRAPH.nodes[gene.node_id]["pos"]
        print(
            f"  [{i}] {gene.sensor.sensor_type.name:<14} "
            f"node={gene.node_id:>4}  "
            f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
            f"pitch={gene.pitch:+4d}°  roll={gene.roll:+4d}°  yaw={gene.yaw:+5d}°"
        )
    ground_pts = debug["ground_cover_points"]
    cyl_pts = debug["cyl_cover_points"]
    print(
        f"  coverage:  S_gnd={len(ground_pts)} cells (green),  "
        f"S_cyl={len(cyl_pts)} cells (blue)"
    )

    mesh = build_robot_surface()
    mesh.visual.face_colors = [110, 110, 110, 80]

    scene_items: list = [mesh]

    for gene in individual:
        node = MOUNTING_GRAPH.nodes[gene.node_id]
        pos = np.array(node["pos"], dtype=float)
        color = SENSOR_COLORS.get(gene.sensor.sensor_type, [200, 200, 200, 230])

        # Sphere at the mounting position
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
        sphere.apply_translation(pos)
        sphere.visual.face_colors = color
        scene_items.append(sphere)

        if show_arrows:
            # Arrow toward the sensor's pointing direction (repo convention:
            # yaw = bearing from +X, positive pitch tilts up).
            yaw_rad = np.radians(gene.yaw)
            pitch_rad = np.radians(gene.pitch)
            direction = np.array(
                [
                    np.cos(pitch_rad) * np.cos(yaw_rad),
                    np.cos(pitch_rad) * np.sin(yaw_rad),
                    np.sin(pitch_rad),
                ]
            )
            arrow_tip = pos + direction * 0.15
            path = trimesh.load_path(np.array([[pos, arrow_tip]]))
            arrow_color = color[:3] + [255]
            path.colors = np.tile(arrow_color, (len(path.entities), 1))
            scene_items.append(path)

    # --- Coverage point clouds (what the fitness counts) ---
    if show_ground and len(ground_pts):
        scene_items.append(
            trimesh.points.PointCloud(ground_pts, colors=[60, 200, 90, 200])
        )
    if show_cyl and len(cyl_pts):
        scene_items.append(
            trimesh.points.PointCloud(cyl_pts, colors=[70, 140, 235, 200])
        )

    # --- Cast rays, colored by what they strike ---
    if show_rays:
        segments: list = []
        seg_colors: list = []
        for sensor_dbg in debug["sensors"]:
            origins = sensor_dbg["ray_origins"]
            endpoints = sensor_dbg["ray_endpoints"]
            categories = sensor_dbg["ray_categories"]
            for o, e, cat in zip(origins, endpoints, categories):
                segments.append([o, e])
                seg_colors.append(RAY_COLORS[int(cat)])
        if segments:
            # Build the path with one Line entity per segment. (trimesh.load_path
            # would merge segments sharing the common sensor origin into multi-
            # vertex entities, breaking the per-entity color mapping.)
            vertices = np.asarray(segments, dtype=float).reshape(-1, 3)
            entities = [Line(np.array([2 * i, 2 * i + 1])) for i in range(len(segments))]
            rays_path = Path3D(entities=entities, vertices=vertices)
            rays_path.colors = np.array(seg_colors, dtype=np.uint8)
            scene_items.append(rays_path)

    scene = trimesh.Scene(scene_items)
    print("\nRendering 3D viewer inline...")

    # Force the GL viewer and explicitly display it in the cell
    display(scene.show(viewer="gl", line_settings={"line_width": 1}))


def visualize_evolution(logbook) -> None:
    import matplotlib

    if not hasattr(matplotlib.rcParams, "_get"):
        matplotlib.rcParams._get = matplotlib.rcParams.get

    # Now you can run your imports and function safely
    import matplotlib.pyplot as plt

    generations = logbook.select("gen")
    average_fitness = logbook.select("avg")
    minimum_fitness = logbook.select("min")
    maximum_fitness = logbook.select("max")

    plt.figure(figsize=(10, 5))
    plt.plot(generations, average_fitness, label="Average fitness", linewidth=2)
    plt.plot(
        generations, minimum_fitness, label="Minimum fitness", linestyle="--", alpha=0.8
    )
    plt.plot(
        generations, maximum_fitness, label="Maximum fitness", linestyle="--", alpha=0.8
    )
    plt.fill_between(generations, minimum_fitness, maximum_fitness, alpha=0.12)
    plt.title("Evolution of Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()
