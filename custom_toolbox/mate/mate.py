import random
from typing import List, Tuple
from config.params import Gene, Individual, Population, MAX_SENSORS_PER_INDIVIDUAL

def cx_variable_length_bounded(ind1: Individual, ind2: Individual) -> Tuple[Individual, Individual]:
    """
    Executes a variable-length one-point crossover while strictly enforcing
    a maximum and minimum sensor limit.
    """
    size1 = len(ind1)
    size2 = len(ind2)
    
    # 1. Edge Case Protection
    if size1 == 0 or size2 == 0:
        return ind1, ind2

    # 2. The Bounded Retry Loop
    # We give the algorithm up to 10 attempts to find valid cut points.
    MAX_ATTEMPTS = 10
    
    for attempt in range(MAX_ATTEMPTS):
        cxpoint1 = random.randint(0, size1)
        cxpoint2 = random.randint(0, size2)

        # Calculate exactly how long the resulting offspring will be
        # Off1 gets Parent 1's head + Parent 2's tail
        len_off1 = cxpoint1 + (size2 - cxpoint2)
        # Off2 gets Parent 2's head + Parent 1's tail
        len_off2 = cxpoint2 + (size1 - cxpoint1)

        # 3. Constraint Validation
        # Check if BOTH offspring will have between 1 and max_sensors
        if (0 < len_off1 <= MAX_SENSORS_PER_INDIVIDUAL) and (0 < len_off2 <= MAX_SENSORS_PER_INDIVIDUAL):
            
            # Valid points found! Execute the actual slice and swap.
            offspring1_genes = ind1[:cxpoint1] + ind2[cxpoint2:]
            offspring2_genes = ind2[:cxpoint2] + ind1[cxpoint1:]
            
            # Modify the parents in-place to satisfy DEAP framework rules
            ind1[:] = offspring1_genes
            ind2[:] = offspring2_genes
            
            return ind1, ind2
            
    # 4. Fallback
    # If no valid cut points were found after 10 tries, we abort the 
    # crossover and return the parents unaltered to protect the population.
    return ind1, ind2