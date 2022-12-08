import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def dry_run():
    """
    Assume undirected graph for now.

    I remember that there were some weights that we had to set?
    Where are they set? Look up in the paper and then the code.

    :return: None
    """
    graph = nx.karate_club_graph()

    random_walk(graph)


def random_walk(graph: nx.Graph):

    nx.draw(graph)
    plt.show()

    A = nx.adjacency_matrix(graph).toarray()

    N = len(A)

    PI = np.ones((N, N))
    PI /= np.sum(PI, axis=1)

    j = np.random.randint(N)

    heatmap = np.zeros(N)

    alpha = 0.5

    for t in range(100):
        """
        p(i, t) - probability that the walker occupies node i at time t

        PI[i, j]:
            - transition matrix of probabilities moving from node j to node i as time t
            - columns must sum to 1
        """
        p = PI @ PI[:, j]

        while j == (i := np.random.choice(range(N), p=p)):
            continue
        heatmap[i] += 1

        print(f"{j} -> {i}")

        div = 0
        for l in range(N):
            div += A[l, j] * np.exp(alpha * p[l])

        for k in range(N):
            PI[k, j] = (A[k, j] * np.exp(alpha * p[k])) / div

        j = i

    print(heatmap)

    nx.draw(graph, node_color=heatmap, cmap='hot')
    plt.show()
