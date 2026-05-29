import random
from enum import Enum

from config.params import Gene, Individual, Population
from config.params import VALID_NODE_IDS, MAX_SENSORS_PER_INDIVIDUAL
from config.graph import MOUNTING_GRAPH
from config.sensors import SENSOR_CATALOG
from custom_toolbox.initialize import initialize

class AttributeMutationType(Enum):
    ANGLES = "angles"
    POSITION = "position"
    HARDWARE = "hardware"

def mutate_sensor_layout(individual: Individual) -> tuple[Individual]:
    """
    Structural Mutation (Add / Drop) - 20% Chance:
    - 50/50 split between Add and Drop
    - Add: If < MAX_SENSORS_PER_INDIVIDUAL, add a new sensor at a random unoccupied node.
    - Drop: If > 1 sensor exists, remove a random sensor from the layout.

    Attribute Mutation (Jitter / Move / Hardware) - 30% Chance per Gene:
    - Applies one of three mutation types to the individual's sensor layout:
        1. ANGLES: Jitter the pitch/roll/yaw of a random sensor within mechanical limits (±90° pitch, ±90° roll, ±180° yaw)
        2. POSITION: Slide a sensor to a graph-adjacent neighbor node while preserving its angles (no-op if all neighbors are occupied)
        3. HARDWARE: Swap the sensor type of a random gene for a different model from the catalog, keeping node and angles the same.
    """
    
    # ---------------------------------------------------------
    # 1. Structural Mutation (Add / Drop) - 20% Chance
    # ---------------------------------------------------------
    if random.random() < 0.20:
        occupied_nodes = set(gene.node_id for gene in individual)
        
        # 50/50 split between Add and Drop
        if random.random() < 0.5 and len(individual) < MAX_SENSORS_PER_INDIVIDUAL:
            available_nodes = list(set(VALID_NODE_IDS) - occupied_nodes)
            if available_nodes:
                new_node = random.choice(available_nodes)
                individual.append(initialize.create_gene(new_node))
                
        elif len(individual) > 1:
            idx = random.randrange(len(individual))
            individual.pop(idx)

    # ---------------------------------------------------------
    # 2. Attribute Mutation (Jitter / Move)
    # ---------------------------------------------------------
    # Evaluate EVERY sensor independently for a chance to mutate
    for gene in individual:
        if random.random() < 0.30: # 30% chance per gene
            
            mutation_type = random.choice(list(AttributeMutationType))
            
            if mutation_type == AttributeMutationType.ANGLES:
                # Your Gaussian micro-adjustments
                gene.pitch = int(round(max(-90, min(90, gene.pitch + random.gauss(0, 5.0)))))
                gene.roll = int(round(max(-90, min(90, gene.roll + random.gauss(0, 5.0)))))
                gene.yaw = int(round(max(-180, min(180, gene.yaw + random.gauss(0, 5.0)))))
                
            elif mutation_type == AttributeMutationType.POSITION:
                # Slide to a graph-adjacent neighbor that is not already occupied.
                occupied_nodes = set(g.node_id for g in individual)
                neighbor_candidates = [
                    n for n in MOUNTING_GRAPH.neighbors(gene.node_id)
                    if n not in occupied_nodes
                ]
                if neighbor_candidates:
                    gene.node_id = random.choice(neighbor_candidates)
                # If all neighbors are occupied or the node is isolated, leave unchanged.

            elif mutation_type == AttributeMutationType.HARDWARE:
                # Swap the sensor out for a different model from the catalog
                gene.sensor = random.choice(list(SENSOR_CATALOG.values()))

    return individual,