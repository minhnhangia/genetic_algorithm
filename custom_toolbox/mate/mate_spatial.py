"""Spatial (front/back) crossover for the LiDAR layout GA.

Where :func:`custom_toolbox.mate.mate.cx_variable_length_bounded` splices parents
by *list index* -- which carries no physical meaning for a placement problem --
this operator recombines parents by *where their sensors sit on the robot*. It is
a specialisation of Matt Ryerkerk's ``Crossover_Spatial`` (metameric EA): a plane
perpendicular to the robot's length axis partitions each parent's sensors into a
front half and a back half, and the halves are swapped across parents.

Geometry
--------
Coordinate convention (shared with the evaluator): ``x = forward, y = left,
z = up``. The mounting graph's x-extent (span ~0.78 m) is the robot's longest
dimension, so **x is the length axis** (y ~0.68 m is width, z ~0.40 m is height).
The cutting plane is perpendicular to x; a sensor is "front" when its mounting
node's x is ``>= split``, else "back".

Rather than a plane fixed at the robot's geometric mid-length, the split is
placed at the **median x of the two parents' combined sensors**, recomputed for
every pair. This makes the cut divide *their* sensors into roughly equal halves,
so both parents tend to contribute to both sides -- avoiding the degenerate case
(common with a fixed plane) where every sensor lands on one side and no real
recombination happens.

Recombination
-------------
Partition each parent into its front and back sensors, then swap the back regions
across parents (equivalently, swap fronts) to form two children::

    child1 = parent1.front + parent2.back
    child2 = parent2.front + parent1.back

So each child carries a *front-of-robot* arrangement from one parent and a
*back-of-robot* arrangement from the other -- spatially coherent blocks, not
arbitrary index slices. Two useful properties fall out:

* **Conservation:** ``len(child1) + len(child2) == len(parent1) + len(parent2)``.
* **Disjointness:** every parent gene ends up in exactly one child, so the two
  children never share a ``Gene`` object. This matches the reference-sharing
  model of the existing one-point operator (the evolution loop deep-clones
  offspring before mating, so the parents themselves are never aliased).

Bounds handling
---------------
Layouts must hold between 1 and ``MAX_SENSORS_PER_INDIVIDUAL`` sensors. The
median split divides the combined sensors evenly but says nothing about how each
*individual* child's count comes out, so a child can still exceed the cap (both
parents' fronts piling up) or come out empty. Because the split is deterministic
for a given pair there is nothing to re-roll, so if either child would fall
outside ``[1, MAX]`` we abort and return the parents unchanged -- mirroring the
fallback in ``cx_variable_length_bounded``. Such pairs simply rely on mutation
this generation and reshuffled pairings the next.
"""

import statistics
from typing import Tuple

from config.params import Gene, Individual, MAX_SENSORS_PER_INDIVIDUAL
from config.graph import MOUNTING_GRAPH

# x = forward is the robot's length axis (longest node-position span). Map each
# mounting node to its x coordinate so a gene's side is a dict lookup.
_LENGTH_AXIS: int = 0  # index into a node's (x, y, z) position
_NODE_AXIS_COORD: dict[int, float] = {
    node_id: float(MOUNTING_GRAPH.nodes[node_id]["pos"][_LENGTH_AXIS])
    for node_id in MOUNTING_GRAPH.nodes()
}


def cx_spatial_front_back(
    ind1: Individual, ind2: Individual
) -> Tuple[Individual, Individual]:
    """Spatial crossover that swaps front/back halves of the robot between parents.

    The cutting plane is placed at the median length-axis (x) coordinate of the
    two parents' combined sensors; each child then inherits one parent's
    front-of-robot sensors and the other parent's back-of-robot sensors (see the
    module docstring for the geometry and the conservation/disjointness
    properties). Children are committed in place per the DEAP convention; if
    either would violate the sensor-count bounds the parents are returned
    unchanged.
    """
    # Edge case: an empty parent has no sensors to partition.
    if len(ind1) == 0 or len(ind2) == 0:
        return ind1, ind2

    # Per-pair split: median x of both parents' combined sensors.
    split = statistics.median(
        [_NODE_AXIS_COORD[gene.node_id] for gene in ind1]
        + [_NODE_AXIS_COORD[gene.node_id] for gene in ind2]
    )

    def is_front(gene: Gene) -> bool:
        return _NODE_AXIS_COORD[gene.node_id] >= split

    p1_front = [gene for gene in ind1 if is_front(gene)]
    p1_back = [gene for gene in ind1 if not is_front(gene)]
    p2_front = [gene for gene in ind2 if is_front(gene)]
    p2_back = [gene for gene in ind2 if not is_front(gene)]

    # Swap back regions across parents (front from one, back from the other).
    child1 = p1_front + p2_back
    child2 = p2_front + p1_back

    # The split is deterministic for this pair, so there is nothing to retry: if
    # either child is out of bounds, abort and leave the parents untouched.
    if not (0 < len(child1) <= MAX_SENSORS_PER_INDIVIDUAL):
        return ind1, ind2
    if not (0 < len(child2) <= MAX_SENSORS_PER_INDIVIDUAL):
        return ind1, ind2

    # Commit in place so DEAP sees the mutated parent objects.
    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2


if __name__ == "__main__":
    # Smoke test: two parents that each straddle the split; confirm the swap.
    from config.params import VALID_NODE_IDS
    from config.sensors import SENSOR_CATALOG, SensorType

    mid = (
        min(_NODE_AXIS_COORD.values()) + max(_NODE_AXIS_COORD.values())
    ) / 2.0
    front_nodes = [n for n in VALID_NODE_IDS if _NODE_AXIS_COORD[n] >= mid]
    back_nodes = [n for n in VALID_NODE_IDS if _NODE_AXIS_COORD[n] < mid]
    sensor = SENSOR_CATALOG[SensorType.LIDAR_16_CH]

    def _g(node_id: int) -> Gene:
        return Gene(sensor=sensor, node_id=node_id, pitch=0, roll=0, yaw=0)

    # Parent 1: one front + one back; Parent 2: one front + one back.
    a = [_g(front_nodes[0]), _g(back_nodes[0])]
    b = [_g(front_nodes[1]), _g(back_nodes[1])]
    print("before:  ind1 nodes =", [g.node_id for g in a],
          " ind2 nodes =", [g.node_id for g in b])
    cx_spatial_front_back(a, b)
    print("after :  ind1 nodes =", [g.node_id for g in a],
          " ind2 nodes =", [g.node_id for g in b])
    # Expect ind1 = [front_nodes[0], back_nodes[1]], ind2 = [front_nodes[1], back_nodes[0]]
