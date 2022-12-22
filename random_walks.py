import networkx as nx
import numpy as np
from graphs import HEAT_ATTRIBUTE


def random_walk_iteration(graph: nx.Graph,
                          A: np.ndarray,
                          PI: np.ndarray,
                          j: int,
                          alpha: int,
                          verbose: bool) -> int:
    """
    p(i, t) - probability that the walker occupies node i at time t

    PI[i, j]:
        - transition matrix of probabilities moving from node j to node i as time t
        - columns must sum to 1

    :returns: index of newly occupied node.
    """
    N = len(A)

    p = PI @ PI[:, j]

    if np.isnan(p).any():
        print(p.sum())
        print(p)

    while j == (i := np.random.choice(range(N), p=p)):
        continue

    div = 0
    for l in range(N):
        div += A[l, j] * np.exp(alpha * p[l])

    for k in range(N):
        PI[k, j] = (A[k, j] * np.exp(alpha * p[k])) / div

    if verbose:
        print(f'{j} -> {i}')

    # update heat
    graph.nodes[i][HEAT_ATTRIBUTE] += 1
    if i in graph[j]:
        graph[j][i][HEAT_ATTRIBUTE] += 1
    else:
        # todo: is this a bug? I guess no..
        if verbose:
            print(f'There is no edge: {j} -> {i}')

    return i
