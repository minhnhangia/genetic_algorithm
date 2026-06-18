"""Phase-0 feasibility gate for the footprint-surrogate approach.

Two checks, runnable as ``python -m surrogate.phase0_gate`` from the project root:

1. **Unit:** a one-gene footprint extracted via :func:`sensor_footprint` is
   bit-identical to what ``CoverageEvaluator.coverage_debug`` reports.
2. **Gate:** the whole approach assumes layout coverage = union of independent
   per-sensor footprints. That drops inter-sensor body occlusion, which can only
   *remove* coverage, so the union OVER-counts. We quantify the gap on random
   multi-sensor layouts (covered-cell error and the resulting fitness error). If
   the gap is small the approximation -- and thus the surrogate -- is sound.

This runs on the *current* robot only (the multi-shape library is a Phase-1
prerequisite); the inter-sensor occlusion physics it gates is robot-agnostic.
"""

from __future__ import annotations

import numpy as np

from custom_toolbox.evaluate.evaluate_fitness_raycast import CoverageEvaluator
from custom_toolbox.initialize import initialize

from .footprints import footprint_flat, sensor_footprint

GATE_THRESHOLD = 0.02  # median covered-cell error fraction to pass


def _unit_check(evaluator: CoverageEvaluator) -> bool:
    """sensor_footprint == coverage_debug grids, bit for bit."""
    from config.params import Gene, VALID_NODE_IDS
    from config.sensors import SENSOR_CATALOG, SensorType

    ok = True
    for st in SensorType:
        sensor = SENSOR_CATALOG[st]
        node = VALID_NODE_IDS[0]
        g, c = sensor_footprint(evaluator, sensor, node, pitch=-10, roll=0, yaw=30)
        dbg = evaluator.coverage_debug(
            [Gene(sensor=sensor, node_id=node, pitch=-10, roll=0, yaw=30)]
        )
        same = np.array_equal(g, dbg["ground_grid"]) and np.array_equal(
            c, dbg["cyl_grid"]
        )
        ok = ok and same
        print(f"  {st.name:13s} footprint == coverage_debug: {same}")
    return ok


def _gate(evaluator: CoverageEvaluator, n_layouts: int, seed: int) -> dict:
    """Union-of-footprints vs true multi-sensor coverage over random layouts."""
    rng = np.random.default_rng(seed)
    scorer = evaluator._scorer
    cell_err, fit_err, n_sensors = [], [], []

    made = 0
    while made < n_layouts:
        layout = initialize.create_individual(list)
        if len(layout) < 2:  # need >=2 sensors to exercise inter-sensor occlusion
            continue
        made += 1
        n_sensors.append(len(layout))

        # True coverage (full layout: includes inter-sensor body occlusion).
        true_fit = evaluator._compute_fitness(layout)[0]
        true_cov = int(evaluator.last_ground_grid.sum()) + int(
            evaluator.last_cyl_grid.sum()
        )

        # Approx coverage: OR of independently-evaluated per-sensor footprints.
        union = None
        for gene in layout:
            g, c = sensor_footprint(
                evaluator, gene.sensor, gene.node_id, gene.pitch, gene.roll, gene.yaw
            )
            flat = footprint_flat(g, c)
            union = flat if union is None else (union | flat)
        approx_cov = int(union.sum())

        cell_err.append((approx_cov - true_cov) / scorer.total_cells)
        fit_err.append(abs(scorer.score(approx_cov, layout) - true_fit))

    cell_err = np.asarray(cell_err)
    fit_err = np.asarray(fit_err)
    return {
        "n_layouts": made,
        "mean_sensors": float(np.mean(n_sensors)),
        "cell_err_median": float(np.median(cell_err)),
        "cell_err_mean": float(np.mean(cell_err)),
        "cell_err_p95": float(np.percentile(cell_err, 95)),
        "cell_err_max": float(np.max(cell_err)),
        "fit_err_median": float(np.median(fit_err)),
        "fit_err_p95": float(np.percentile(fit_err, 95)),
        "fit_err_max": float(np.max(fit_err)),
    }


def main(n_layouts: int = 5000, seed: int = 0) -> None:
    evaluator = CoverageEvaluator()
    print(
        f"grid: {evaluator._scorer.total_cells} cells "
        f"(ground {evaluator.ground.n_cells} + cyl {evaluator.cylinder.n_cells})\n"
    )

    print("== Unit: footprint extraction ==")
    unit_ok = _unit_check(evaluator)

    print(f"\n== Gate: union-of-footprints vs true coverage ({n_layouts} layouts) ==")
    r = _gate(evaluator, n_layouts, seed)
    print(f"  layouts={r['n_layouts']}  mean sensors/layout={r['mean_sensors']:.2f}")
    print(f"  covered-cell error (union overcounts):")
    print(
        f"    median={r['cell_err_median']:.4f}  mean={r['cell_err_mean']:.4f}"
        f"  p95={r['cell_err_p95']:.4f}  max={r['cell_err_max']:.4f}"
    )
    print(f"  fitness error:")
    print(
        f"    median={r['fit_err_median']:.5f}  p95={r['fit_err_p95']:.5f}"
        f"  max={r['fit_err_max']:.5f}"
    )

    passed = unit_ok and r["cell_err_median"] < GATE_THRESHOLD
    print(
        f"\nVERDICT: {'GO' if passed else 'NO-GO'} "
        f"(unit={'ok' if unit_ok else 'FAIL'}, "
        f"median cell error {r['cell_err_median']:.4f} vs threshold {GATE_THRESHOLD})"
    )
    if not passed and unit_ok:
        print(
            "  -> approximation too lossy; add a pairwise inter-sensor occlusion "
            "correction before the surrogate."
        )


if __name__ == "__main__":
    main()
