import math

from config.params import Gene, Population, Individual

def evaluate_individual(individual: Individual) -> tuple[float]:
    """
    Evaluates a LiDAR layout based on spatial coverage, critical redundancy, 
    orientation viability, and cost.
    """
    # ---------------------------------------------------------
    # 1. Weights & Config
    # ---------------------------------------------------------
    W_COV = 0.45        # Weight for 360-degree coverage
    W_CRIT = 0.35       # Weight for redundant coverage in critical zones
    W_COST = 0.20       # Weight for financial cost
    
    MAX_BUDGET = 5000.0
    OPTIMAL_PITCH = -10 # Degrees. Assuming pointing slightly down is ideal for ground robots
    
    # Critical Zone Definition (e.g., front of the robot: -45 to +45 degrees)
    CRIT_ZONE_START = 315
    CRIT_ZONE_END = 45

    # ---------------------------------------------------------
    # 2. Financial Penalty
    # ---------------------------------------------------------
    total_cost = sum(gene.sensor.price for gene in individual)
    c_norm = min(total_cost / MAX_BUDGET, 1.0)

    # ---------------------------------------------------------
    # 3. Spatial Coverage Mapping (The 360 Array)
    # ---------------------------------------------------------
    # Create an array of 360 zeros. Each index represents 1 degree around the robot.
    # The value at each index will be the number of sensors covering that degree.
    coverage_map = [0] * 360
    
    for gene in individual:
        # Penalize extreme pitch/roll angles. 
        # A sensor pointing straight into the sky (+90 pitch) provides 0 useful ground coverage.
        # We simulate this by scaling down its effective FOV.
        pitch_deviation = abs(gene.pitch - OPTIMAL_PITCH)
        orientation_penalty = math.cos(math.radians(pitch_deviation))
        
        # Ensure penalty doesn't invert the FOV; clamp to [0, 1]
        orientation_penalty = max(0.0, min(1.0, orientation_penalty))
        effective_fov = gene.sensor.fov_horizontal_deg * orientation_penalty
        
        # Calculate coverage sweep
        half_fov = effective_fov / 2.0
        start_angle = int(gene.yaw - half_fov)
        end_angle = int(gene.yaw + half_fov)
        
        # Fill the coverage map, handling 360-degree wrap-around
        for angle in range(start_angle, end_angle):
            normalized_angle = (angle + 360) % 360
            coverage_map[normalized_angle] += 1

    # ---------------------------------------------------------
    # 4. Metric Calculations
    # ---------------------------------------------------------
    
    # M_cov: Percentage of the 360 environment covered by at least 1 sensor
    covered_bins = sum(1 for hits in coverage_map if hits > 0)
    m_cov = covered_bins / 360.0 

    # M_crit: Redundancy in the critical zone
    # We want at least 2 sensors covering the front of the robot for redundancy.
    crit_bins_total = 0
    crit_bins_redundant = 0
    
    # Iterate through the critical zone (handling wrap-around)
    angle = CRIT_ZONE_START
    while angle != CRIT_ZONE_END:
        crit_bins_total += 1
        if coverage_map[angle] >= 2:  # Redundant if 2 or more hits
            crit_bins_redundant += 1
        angle = (angle + 1) % 360
        
    m_crit = crit_bins_redundant / float(crit_bins_total) if crit_bins_total > 0 else 0.0

    # ---------------------------------------------------------
    # 5. Final Aggregation
    # ---------------------------------------------------------
    fitness_score = (W_COV * m_cov) + (W_CRIT * m_crit) - (W_COST * c_norm)
    
    # Ensure fitness doesn't drop below 0 (for standard roulette wheel selection)
    fitness_score = max(0.0, fitness_score)
    
    # DEAP STRICT REQUIREMENT: Must return a tuple, even for a single objective.
    # The comma is mandatory!
    return (fitness_score,)
