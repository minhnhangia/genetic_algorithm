"""Side-by-side visual comparison of two (or more) GA runs.

Consumes the :class:`~utils.ga_run.RunResult` objects produced by
``utils.ga_run.run_evolution`` -- typically one per selection strategy -- and
overlays / juxtaposes them so the effect of the selection operator is visible:

* :func:`compare_evolution` -- overall best & average fitness curves per run.
* :func:`compare_best_per_length` -- best layout found at each sensor count,
  per run (table + grouped bars). "Which strategy wins at each count?"
* :func:`compare_length_distribution` -- population composition over generations,
  one panel per run. Shows length niching keeping every sensor count alive vs a
  tournament collapsing toward a single count.
* :func:`compare_per_length_evolution` -- per sensor count, both runs overlaid.

Each function takes ``results``: a mapping ``{label: RunResult}`` (e.g. an
ordinary dict, insertion-ordered). Colours are assigned per label and kept
consistent across all four plots.
"""

from __future__ import annotations

import math
from typing import Mapping

from IPython.display import HTML, display

from utils.ga_run import RunResult

# Distinct, consistent colour per run label (assigned in iteration order).
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def _plt():
    """Import matplotlib.pyplot with the repo's rcParams shim applied."""
    import matplotlib

    if not hasattr(matplotlib.rcParams, "_get"):
        matplotlib.rcParams._get = matplotlib.rcParams.get
    import matplotlib.pyplot as plt

    return plt


def _colors(results: Mapping[str, RunResult]) -> dict[str, str]:
    return {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(results)}


def compare_evolution(results: Mapping[str, RunResult]) -> None:
    """Overlay overall best (solid) and average (dashed) fitness per run."""
    plt = _plt()
    colors = _colors(results)

    plt.figure(figsize=(10, 5.5))
    for label, res in results.items():
        gens = res.logbook.select("gen")
        gmax = res.logbook.select("max")
        gavg = res.logbook.select("avg")
        c = colors[label]
        plt.plot(gens, gmax, color=c, linewidth=2, label=f"{label} — best")
        plt.plot(gens, gavg, color=c, linewidth=1.5, linestyle="--", alpha=0.7,
                 label=f"{label} — avg")

    plt.title("Fitness over generations by selection strategy")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_best_per_length(
    results: Mapping[str, RunResult], evaluator=None
) -> None:
    """Compare the best layout found at each sensor count, across runs.

    Renders an HTML table (best fitness and total cost per count per run; plus
    coverage if an ``evaluator`` is supplied) and a grouped bar chart of best
    fitness vs sensor count.
    """
    plt = _plt()
    colors = _colors(results)

    # Union of sensor counts seen by any run.
    all_lengths = sorted(
        {length for res in results.values() for length in res.best_per_length.lengths}
    )
    if not all_lengths:
        display(HTML("<em>No per-length bests recorded in any run.</em>"))
        return

    def _cell(res: RunResult, length: int) -> str:
        bpl = res.best_per_length
        if length not in bpl:
            return "<td style='text-align:center; color:#8c959f;'>—</td>"
        ind = bpl[length]
        fit = ind.fitness.values[0]
        cost = sum(g.sensor.price for g in ind)
        cov = ""
        if evaluator is not None:
            dbg = evaluator.coverage_debug(ind)
            covered = int(dbg["ground_grid"].sum()) + int(dbg["cyl_grid"].sum())
            total = int(dbg["ground_grid"].size) + int(dbg["cyl_grid"].size)
            cov = f"<br><span style='color:#57606a;'>{covered/total:.1%} cov</span>"
        return (
            f"<td style='text-align:right;'>{fit:.4f}"
            f"<br><span style='color:#57606a;'>${cost:,.0f}</span>{cov}</td>"
        )

    head = "".join(f"<th>{label}</th>" for label in results)
    body = "".join(
        f"<tr><td style='text-align:center; font-weight:700;'>{length}</td>"
        + "".join(_cell(res, length) for res in results.values())
        + "</tr>"
        for length in all_lengths
    )
    display(HTML(f"""
    <style>
      .cmp-table {{ border-collapse: collapse; font-family: Arial, sans-serif;
                    font-size: 14px; margin-bottom: 12px; }}
      .cmp-table th, .cmp-table td {{ border: 1px solid #d0d7de; padding: 8px 12px; }}
      .cmp-table th {{ background: #f6f8fa; text-align: center; }}
    </style>
    <table class='cmp-table'>
      <tr><th>Sensors</th>{head}</tr>
      {body}
    </table>
    """))

    # Grouped bars: best fitness per count per run.
    n = len(results)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, res) in enumerate(results.items()):
        bpl = res.best_per_length
        xs = [length + (i - (n - 1) / 2) * width for length in all_lengths]
        ys = [bpl[length].fitness.values[0] if length in bpl else 0.0
              for length in all_lengths]
        bars = ax.bar(xs, ys, width=width, color=colors[label], alpha=0.8, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(all_lengths)
    ax.set_xlabel("Number of sensors")
    ax.set_ylabel("Best fitness found")
    ax.set_title("Best layout per sensor count by selection strategy")
    ax.legend()
    fig.tight_layout()
    plt.show()


def compare_length_distribution(
    results: Mapping[str, RunResult], *, normalize: bool = True
) -> None:
    """Population composition by sensor count over generations, one panel per run.

    Each panel is a stacked-area chart (bands = sensor counts). The contrast
    between panels is the headline result: niching holds every count's band open
    while a tournament tends to let one band swallow the population.
    """
    plt = _plt()

    if not results:
        return

    # Shared band ordering / colours across panels so they're directly comparable.
    all_lengths = sorted(
        {length for res in results.values() for length in res.per_length_evolution.lengths}
    )
    if not all_lengths:
        display(HTML("<em>No per-length evolution recorded in any run.</em>"))
        return
    band_colors = {
        length: _PALETTE[i % len(_PALETTE)] for i, length in enumerate(all_lengths)
    }

    fig, axes = plt.subplots(
        1, len(results), figsize=(7 * len(results), 5), squeeze=False, sharey=normalize
    )
    for ax, (label, res) in zip(axes[0], results.items()):
        ple = res.per_length_evolution
        gens = ple.generations
        counts = []
        for length in all_lengths:
            _, vals = ple.series(length, "count")
            counts.append([0.0 if math.isnan(v) else v for v in vals])

        if normalize:
            totals = [sum(col) for col in zip(*counts)]
            counts = [[(v / t if t else 0.0) for v, t in zip(row, totals)]
                      for row in counts]

        ax.stackplot(
            gens, counts,
            labels=[f"{length} sensor{'s' if length != 1 else ''}" for length in all_lengths],
            colors=[band_colors[length] for length in all_lengths],
            alpha=0.85,
        )
        ax.set_title(label)
        ax.set_xlabel("Generation")
        if gens:
            ax.set_xlim(min(gens), max(gens))
        if normalize:
            ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.2)

    axes[0][0].set_ylabel("Fraction of population" if normalize else "Number of individuals")
    axes[0][-1].legend(title="Sensor count", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle(
        "Population composition by sensor count"
        + (" (normalized)" if normalize else ""),
        fontsize=13,
    )
    fig.tight_layout()
    plt.show()


def compare_per_length_evolution(
    results: Mapping[str, RunResult], metric: str = "max"
) -> None:
    """Per sensor count, overlay each run's ``metric`` fitness over generations.

    One subplot per sensor count (small multiples), so you can see, e.g., whether
    niching keeps improving the best 4-sensor layout where a tournament lets it
    go extinct (its line breaks at the extinction gap).
    """
    plt = _plt()
    colors = _colors(results)

    all_lengths = sorted(
        {length for res in results.values() for length in res.per_length_evolution.lengths}
    )
    if not all_lengths:
        display(HTML("<em>No per-length evolution recorded in any run.</em>"))
        return

    metric_label = {"max": "best", "avg": "average", "min": "minimum",
                    "count": "individual count"}.get(metric, metric)

    ncol = min(len(all_lengths), 2)
    nrow = math.ceil(len(all_lengths) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4 * nrow), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, length in zip(flat, all_lengths):
        for label, res in results.items():
            gens, vals = res.per_length_evolution.series(length, metric)
            ax.plot(gens, vals, color=colors[label], linewidth=2, label=label)
        ax.set_title(f"{length} sensor{'s' if length != 1 else ''}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness" if metric != "count" else "Individuals")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    # Hide any unused axes.
    for ax in flat[len(all_lengths):]:
        ax.set_visible(False)

    fig.suptitle(f"Per-sensor-count {metric_label} fitness by strategy", fontsize=13)
    fig.tight_layout()
    plt.show()
