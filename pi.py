import networkx as nx
import numpy as np


def uniform(graph: nx.Graph) -> np.ndarray:
    N = len(graph.nodes)
    return np.full((N, N), 1 / N)


def proportional_to_node_degree(graph: nx.Graph) -> np.ndarray:
    N = len(graph.nodes)
    degrees = [deg + 1 for _, deg in graph.degree]
    total = sum(degrees)
    p=np.array([deg / total for deg in degrees])

    PI = np.tile(p, (N, 1)).T
    assert PI.shape == (N, N)

    return PI

def inversely_proportional_to_node_degree(graph: nx.Graph) -> np.ndarray:
    N = len(graph.nodes)
    degrees = [1 / (deg + 1) for _, deg in graph.degree]
    total = sum(degrees)
    p=np.array([deg / total for deg in degrees])

    PI = np.tile(p, (N, 1)).T
    assert PI.shape == (N, N)

    return PI
