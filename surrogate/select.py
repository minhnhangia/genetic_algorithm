"""Greedy submodular layout selection over a footprint table.

A *candidate* is a ``(node, sensor_type, orientation)`` with its flat footprint
mask. Greedy repeatedly adds the candidate with the best **marginal fitness gain**
(coverage blend minus the new sensor's cost), enforcing unique mounting nodes and
the sensor-count cap, and stops when no addition improves fitness -- which also
chooses the sensor *count*. Coverage is submodular and per-sensor cost is modular,
so the objective is submodular and greedy has the standard ``1 - 1/e`` flavour.

The footprint table is built here from the **true** raycast evaluator -- the exact
no-ML baseline and the upper bound the learned surrogate will be measured against.
The surrogate will later supply predicted footprints through the same ``Candidate``
interface, leaving :func:`greedy_select` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.params import Gene, MAX_SENSORS_PER_INDIVIDUAL, VALID_NODE_IDS
from config.sensors import SENSOR_CATALOG, Sensor
from custom_toolbox.evaluate.evaluate_fitness_raycast import CoverageEvaluator
from custom_toolbox.evaluate.scoring import FitnessScorer
from custom_toolbox.utils.utils import select_spread_nodes

from .footprints import footprint_flat, is_omnidirectional, sensor_footprint


@dataclass
class Candidate:
    """One placeable sensor pose plus its precomputed footprint mask."""

    node_id: int
    sensor: Sensor
    pitch: int
    roll: int
    yaw: int
    mask: np.ndarray  # flat (n_cells,) boolean footprint

    def gene(self) -> Gene:
        return Gene(
            sensor=self.sensor,
            node_id=self.node_id,
            pitch=self.pitch,
            roll=self.roll,
            yaw=self.yaw,
        )


def default_orientations(sensor: Sensor) -> list[tuple[int, int, int]]:
    """Coarse ``(pitch, roll, yaw)`` grid per sensor type.

    Omnidirectional sensors vary the spin axis only (level plus a few tilt
    magnitudes/azimuths; roll redundant -> 0). Directional sensors sweep the bore
    aim (yaw) at a couple of pitches. Kept small so the footprint table stays cheap;
    greedy then aims each sensor to complement the others.
    """
    if is_omnidirectional(sensor):
        oris = [(0, 0, 0)]  # level is often optimal at central mounts
        for pitch in (-25, 25):
            for yaw in (0, 90, 180, -90):
                oris.append((pitch, 0, yaw))
        return oris
    oris = []
    for pitch in (0, -15, 10):
        for yaw in (0, 60, 120, 180, -120, -60):
            oris.append((pitch, 0, yaw))
    return oris


def build_candidates(
    evaluator: CoverageEvaluator,
    node_ids,
    sensors: list[Sensor] | None = None,
    orientations_fn=default_orientations,
) -> list[Candidate]:
    """Precompute the true footprint for every ``(node, sensor, orientation)``."""
    sensors = sensors or list(SENSOR_CATALOG.values())
    cands: list[Candidate] = []
    for node in node_ids:
        for sensor in sensors:
            for pitch, roll, yaw in orientations_fn(sensor):
                g, c = sensor_footprint(evaluator, sensor, node, pitch, roll, yaw)
                cands.append(
                    Candidate(node, sensor, pitch, roll, yaw, footprint_flat(g, c))
                )
    return cands


def greedy_select(
    candidates: list[Candidate],
    scorer: FitnessScorer,
    max_sensors: int = MAX_SENSORS_PER_INDIVIDUAL,
) -> tuple[list[Candidate], float]:
    """Greedily build a layout maximising fitness; returns ``(selected, fitness)``.

    Adds the best marginal-fitness candidate each round (one sensor per node),
    stopping at ``max_sensors`` or when no candidate improves fitness.
    """
    selected: list[Candidate] = []
    union: np.ndarray | None = None
    used_nodes: set[int] = set()
    cur_fit = 0.0

    while len(selected) < max_sensors:
        best: Candidate | None = None
        best_fit = cur_fit
        best_union: np.ndarray | None = None
        base_genes = [c.gene() for c in selected]

        for cand in candidates:
            if cand.node_id in used_nodes:
                continue
            new_union = cand.mask if union is None else (union | cand.mask)
            fit = scorer.score(int(new_union.sum()), base_genes + [cand.gene()])
            if fit > best_fit:
                best, best_fit, best_union = cand, fit, new_union

        if best is None:  # no positive marginal gain -> optimal count reached
            break
        selected.append(best)
        union = best_union
        used_nodes.add(best.node_id)
        cur_fit = best_fit

    return selected, cur_fit


def optimize_layout(
    evaluator: CoverageEvaluator,
    n_candidate_nodes: int = 80,
    seed: int = 0,
    min_separation_m: float = 0.3,
) -> tuple[list[Gene], float]:
    """End-to-end greedy optimisation on the current robot.

    Picks a spread set of candidate mounting nodes, builds the true footprint table,
    and greedily selects a layout. Returns ``(layout_genes, greedy_fitness)``; the
    caller should verify with ``evaluator.evaluate_individual`` (near-identical by
    the Phase-0 union approximation).
    """
    import random

    random.seed(seed)
    node_ids = select_spread_nodes(n_candidate_nodes, min_separation_m)
    candidates = build_candidates(evaluator, node_ids)
    selected, fit = greedy_select(candidates, scorer=evaluator._scorer)
    return [c.gene() for c in selected], fit
