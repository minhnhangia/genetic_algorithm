"""Per-sensor coverage footprint surrogate + greedy layout selection.

A learning-based amortization of the expensive raycast coverage evaluator across
many robot shapes. The core object is a single sensor's covered-cell footprint on
the shared (ground + cylinder) evaluation grid; layout coverage is recomposed as
the union of footprints and the cost term is computed exactly. See the project
plan for the full rationale and phasing.
"""
