from IPython.display import HTML, display

from params.params import SensorType

def visualize_population(population, max_display=5):
    sensor_colors = {
        SensorType.LIDAR_16_CH.value: "#1f77b4",
        SensorType.LIDAR_32_CH.value: "#ff7f0e",
        SensorType.SOLID_STATE.value: "#2ca02c",
    }

    visible_population = population[:max_display]
    max_genes = max(len(individual) for individual in visible_population)
    rows = []

    for index, individual in enumerate(visible_population, start=1):
        cells = [f'<th>Individual {index}</th>']

        for gene in individual:
            color = sensor_colors[gene.sensor_type.value]
            cells.append(
                f'<td style="border-left: 6px solid {color}; background: {color}22;">'
                f'<div style="font-weight: 700; margin-bottom: 4px;">{gene.sensor_type.name}</div>'
                f'<div>node {gene.node_id}</div>'
                f'<div>pitch {gene.pitch}, roll {gene.roll}</div>'
                f'</td>'
            )

        while len(cells) < max_genes + 1:
            cells.append('<td class="empty">&mdash;</td>')

        rows.append('<tr>' + ''.join(cells) + '</tr>')

    header = '<tr><th>Individual</th>' + ''.join(f'<th>Sensor {i}</th>' for i in range(1, max_genes + 1)) + '</tr>'
    summary = ""

    if len(population) > max_display:
        summary = (
            f'<div style="margin: 0 0 12px 0; color: #57606a; font-size: 13px;">'
            f'Showing first {max_display} of {len(population)} individuals.'
            f'</div>'
        )

    html = f'''
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
    '''

    display(HTML(html))