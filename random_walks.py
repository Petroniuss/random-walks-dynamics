import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def dry_run():
    """
    Assume undirected graph for now.

    I remember that there were some weights that we had to set?
    Where are they set? Look up in the paper and then in the code.

    :return: None
    """
    graph = nx.karate_club_graph()

    random_walk(graph, verbose=True)


def random_walk(graph: nx.Graph,
                alpha: int = 0.5,
                verbose: bool = False):
    """
    it'd be nice if we could yield some results to animation
    so that an update function could control the progress of the algorithm!
    """
    nx.draw(graph)
    plt.show()

    A = nx.adjacency_matrix(graph).toarray()

    N = len(A)

    PI = np.ones((N, N))
    PI /= np.sum(PI, axis=1)

    j = np.random.randint(N)

    node_heatmap = np.zeros(shape=N)
    edge_heatmap = np.zeros(shape=(N, N))

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

        # assumes undirected graph
        edge_heatmap[min(j, i)][max(j, i)] += 1
        node_heatmap[i] += 1

        if verbose:
            print(f'{t}. {j} -> {i}')

        div = 0
        for l in range(N):
            div += A[l, j] * np.exp(alpha * p[l])

        for k in range(N):
            PI[k, j] = (A[k, j] * np.exp(alpha * p[k])) / div

        j = i

    if verbose:
        print(node_heatmap)
        print(edge_heatmap)

    edges = graph.edges()
    for u, v in edges:
        graph[u][v]['weight'] = (1 + edge_heatmap[u][v]) * 2.
        graph[u][v]['color'] = 1 + edge_heatmap[u][v]

    edge_colors = nx.get_edge_attributes(graph,'color').values()
    edge_weights = nx.get_edge_attributes(graph, 'weight').values()

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos,
            node_color=node_heatmap,
            edge_color=edge_colors,
            width=list(edge_weights))
    plt.show()
