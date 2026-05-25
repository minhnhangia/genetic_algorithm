import random

from config.params import Gene, VALID_NODE_IDS
from custom_toolbox.initialize import initialize  

def mutate_sensor_layout(individual):
    """
    Applies one of the three mutations based on probability.
    - 10% chance: DROP a random sensor (if >1 sensor exists)
    - 10% chance: ADD a new sensor at an unoccupied node (if <4 sensors exist)
    - 80% chance: JITTER the pitch/roll of a random sensor within mechanical limits (±90° pitch, ±90° roll)
    """
    mutation_choice = random.random()
    occupied_nodes = set(gene.node_id for gene in individual)
    
    if mutation_choice < 0.10 and len(individual) > 1:
        # DROP MUTAION: Pop a random sensor (safeguarded > 1)
        idx = random.randrange(len(individual))
        individual.pop(idx)
        
    elif 0.10 <= mutation_choice < 0.20 and len(individual) < 4:
        # ADD MUTATION: Find an empty node and add
        available_nodes = list(set(VALID_NODE_IDS) - occupied_nodes)
        if available_nodes:
            new_node = random.choice(available_nodes)
            individual.append(initialize.create_gene(new_node))
            
    else:
        # JITTER MUTATION: Tweak the pitch/roll respecting FLU constraints
        target_gene = random.choice(individual)
        
        # Add Gaussian noise (mean 0, std dev 5 degrees)
        target_gene.pitch += int(round(random.gauss(0, 5.0)))
        target_gene.roll += int(round(random.gauss(0, 5.0)))
        
        # Clip to mechanical bounds
        target_gene.pitch = max(-90.0, min(90.0, target_gene.pitch))
        target_gene.roll = max(-90.0, min(90.0, target_gene.roll))
        
    return individual,