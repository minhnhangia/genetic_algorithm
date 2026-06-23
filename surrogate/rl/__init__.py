"""Deep-RL constructive placement policy with a surrogate reward (research track).

A GNN policy that places sensors sequentially on a robot's mounting graph, trained
across the robot fleet (hold-out by robot) using the learned footprint surrogate as
a fast reward model -- so training never raycasts. See the project plan for the
contribution framing (cross-robot zero-shot placement + surrogate-as-reward).
"""
