# Genetic Algorithm for Sensor Layout Optimization

Notebook-driven prototype for evolving sensor layouts with [DEAP](https://deap.readthedocs.io/). The current workflow explores how a layout of sensors, node placements, and orientations can be optimized with a genetic algorithm.

## What this project does

The main notebook, [main.ipynb](main.ipynb), wires together the full evolutionary loop:

- builds DEAP individuals from `Gene` objects
- seeds the initial population with a few hand-crafted layouts
- evaluates each layout with a mock fitness function
- applies tournament selection, bounded crossover, and custom mutation
- tracks evolution statistics with DEAP `Logbook` and `HallOfFame`
- visualizes the population and fitness history

Each gene represents one mounted sensor with:

- sensor type and spec data
- node ID on the platform
- pitch, roll, and yaw angles

## Why it is useful

This repo is a compact playground for sensor-placement optimization. It is useful if you want to:

- experiment with GA operators for variable-length layouts
- compare seeded individuals against random initialization
- explore tradeoffs between coverage, redundancy, and cost
- visualize how a population changes over generations

The current evaluator is intentionally mocked, so the code is best treated as a working research scaffold rather than a production optimizer.

## Project structure

- [main.ipynb](main.ipynb) - notebook entrypoint and evolutionary loop
- [config/params.py](config/params.py) - gene model, type aliases, and GA constants
- [config/sensors.py](config/sensors.py) - sensor catalog and specs
- [config/seeding.py](config/seeding.py) - expert seed layouts for the starting population
- [custom_toolbox/initialize/initialize.py](custom_toolbox/initialize/initialize.py) - individual and population creation helpers
- [custom_toolbox/evaluate/evaluate_fitness_mock.py](custom_toolbox/evaluate/evaluate_fitness_mock.py) - mock fitness function for coverage, redundancy, and cost
- [custom_toolbox/mate/mate.py](custom_toolbox/mate/mate.py) - bounded variable-length crossover
- [custom_toolbox/mutate/mutate.py](custom_toolbox/mutate/mutate.py) - add/drop, position, angle, and hardware mutation
- [utils/visualization.py](utils/visualization.py) - population table and fitness plotting helpers

## Getting started

### Prerequisites

- Python 3.10 or newer
- Jupyter or VS Code notebook support

### Install dependencies

The repository does not currently include a dependency lockfile, so install the runtime packages manually:

```bash
pip install deap numpy matplotlib ipython jupyter
```

### Run the notebook

Open [main.ipynb](main.ipynb) in Jupyter or VS Code and run the cells in order. The notebook will:

1. create the DEAP fitness and individual classes
2. build a seeded population
3. register evaluation, selection, mutation, and mating functions
4. execute the generation loop
5. render the population and fitness charts

### Example usage

The notebook is the main entrypoint, but the modules can also be reused from Python code:

```python
from deap import base, creator, tools
from custom_toolbox.initialize.initialize import create_individual
from custom_toolbox.evaluate.evaluate_fitness_mock import evaluate_individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", create_individual, creator.Individual)
toolbox.register("evaluate", evaluate_individual)
```

## Help and documentation

- Read the notebook first: [main.ipynb](main.ipynb)
- Inspect the sensor model and GA constants in [config/params.py](config/params.py)
- Review the fitness assumptions in [custom_toolbox/evaluate/evaluate_fitness_mock.py](custom_toolbox/evaluate/evaluate_fitness_mock.py)
- Check the DEAP docs for operator behavior and population management: [DEAP documentation](https://deap.readthedocs.io/)
