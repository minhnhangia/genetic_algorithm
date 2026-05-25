def evaluate_individual(individual):
    """
    Mock evaluation function for the modular ground robot LiDAR layout.
    Returns a single-item tuple representing the fitness score to be maximized.
    """
    # ---------------------------------------------------------
    # 1. State Weights & Constraints
    # ---------------------------------------------------------
    # These dictate the robot's current priorities. 
    W1_CRIT = 0.4       # Weight for Critical Redundancy (Front/Braking path)
    W2_COV = 0.4        # Weight for 360-degree Basin Coverage
    W3_COST = 0.2       # Weight for Financial Penalty
    MAX_BUDGET = 5000.0 # Absolute budget cap (C_max) for normalization

    # ---------------------------------------------------------
    # 2. Calculate Financial Penalty (C_norm)
    # ---------------------------------------------------------
    # Sum the price property from the SENSOR_CATALOG via the gene's sensor reference
    total_cost = sum(gene.sensor.price for gene in individual)
    
    # Normalize against the max budget to keep the penalty bounded between 0.0 and 1.0
    c_norm = min(total_cost / MAX_BUDGET, 1.0)

    # ---------------------------------------------------------
    # 3. Mock the Geometric Raycasting (Coverage & Redundancy)
    # ---------------------------------------------------------
    # In production, this block is replaced by your headless raycaster.
    # It will project vectors from each node onto the S_gnd and S_cyl basins.
    
    # Mock M_cov (General Coverage): bounded between [0, 1]
    # Simulating coverage by summing horizontal FOVs. 
    # (Dividing by 400.0 as a mock threshold where diminishing returns hit)
    total_fov = sum(gene.sensor.fov_horizontal_deg for gene in individual)
    m_cov = min(total_fov / 400.0, 1.0) 
    
    # Mock M_crit (Critical Redundancy): bounded between [0, 1]
    # Simulating that more active sensors generally yield higher overlap in critical zones.
    num_sensors = len(individual)
    m_crit = min((num_sensors - 1) / 3.0, 1.0) if num_sensors > 1 else 0.0

    # ---------------------------------------------------------
    # 4. Aggregated Fitness Calculation
    # ---------------------------------------------------------
    fitness_score = (W1_CRIT * m_crit) + (W2_COV * m_cov) - (W3_COST * c_norm)
    
    # DEAP STRICT REQUIREMENT: Must return a tuple, even for a single objective.
    # The comma is mandatory!
    return (fitness_score,)
