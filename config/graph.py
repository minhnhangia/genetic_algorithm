import pickle
import pathlib

import networkx as nx

# NOTE: pickle.load is unsafe against adversarially crafted files.
# Acceptable here since this file is generated and consumed by the same local environment.
GRAPH_PATH = pathlib.Path(__file__).parent.parent / "data" / "mounting_graph.pkl"


def _load_graph(path: pathlib.Path) -> nx.Graph:
    if not path.exists():
        raise FileNotFoundError(
            f"Mounting graph not found at '{path}'.\n"
            "Run `python generate_mounting_graph.py` first to generate and save it."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


MOUNTING_GRAPH: nx.Graph = _load_graph(GRAPH_PATH)
