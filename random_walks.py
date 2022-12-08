import time
from dataclasses import dataclass

import matplotlib
from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import PathCollection, LineCollection

matplotlib.use("TkAgg")

HEAT_ATTRIBUTE = 'heat'
CMAP = 'hot'

plt.set_cmap(CMAP)


def dry_run_animation():
    """
    Assume undirected graph for now.

    I remember that there were some weights that we had to set?
    Where are they set? Look up in the paper and then in the code.

    :return: None
    """
    graph = nx.karate_club_graph()
    animate_random_walk(graph)


@dataclass
class AnimationIterationArgs:
    graph: nx.Graph
    drawn_nodes: PathCollection
    drawn_edges: LineCollection
    A: np.ndarray
    PI: np.ndarray
    j: int
    alpha: int
    verbose: bool


def animation_update(frame: int, args: AnimationIterationArgs):
    """
    Called once per frame.
    """
    print(f'{frame}')
    new_j = random_walk_iteration(
        graph=args.graph,
        A=args.A,
        PI=args.PI,
        j=args.j,
        alpha=args.alpha,
        verbose=args.verbose,
    )

    args.j = new_j

    edge_weights = nx.get_edge_attributes(args.graph, HEAT_ATTRIBUTE).values()
    edge_colors = nx.get_edge_attributes(args.graph, HEAT_ATTRIBUTE).values()
    node_colors = nx.get_node_attributes(args.graph, HEAT_ATTRIBUTE).values()

    # move this to the algorithm.
    args.drawn_nodes.set_color(normalize_colors(node_colors))
    args.drawn_edges.set_color(normalize_colors(edge_colors))
    args.drawn_edges.set_linewidth(list(edge_weights))

    # update jth width.


def normalize_colors(colors):
    """
    Works almost like nx.draw :)
    """
    cmap = plt.get_cmap()
    edge_vmin = min(colors)
    edge_vmax = max(colors)
    color_normal = matplotlib.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    return [cmap(color_normal(e)) for e in colors]


def animate_random_walk(graph: nx.Graph,
                        alpha: int = 0.5,
                        n_iters: int = 100,
                        verbose: bool = False):
    # algorithm initial state
    A = nx.adjacency_matrix(graph).toarray()
    N = len(A)

    PI = np.ones((N, N))
    PI /= np.sum(PI, axis=1)

    j = np.random.randint(N)

    # animation initial state
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    pos = nx.spring_layout(graph)
    drawn_nodes = nx.draw_networkx_nodes(graph, pos)
    drawn_edges = nx.draw_networkx_edges(graph, pos)

    nx.set_edge_attributes(graph, 1, HEAT_ATTRIBUTE)
    nx.set_node_attributes(graph, 1, HEAT_ATTRIBUTE)

    edge_weights = nx.get_edge_attributes(graph, HEAT_ATTRIBUTE).values()
    edge_colors = nx.get_edge_attributes(graph, HEAT_ATTRIBUTE).values()
    node_colors = nx.get_node_attributes(graph, HEAT_ATTRIBUTE).values()

    # move this to the algorithm.
    drawn_nodes.set_color(normalize_colors(node_colors))
    drawn_edges.set_color(normalize_colors(edge_colors))
    drawn_edges.set_linewidth(list(edge_weights))

    args = AnimationIterationArgs(
        graph=graph,
        drawn_nodes=drawn_nodes,
        drawn_edges=drawn_edges,
        A=A,
        PI=PI,
        j=j,
        alpha=alpha,
        verbose=verbose
    )

    anim = FuncAnimation(
        fig, animation_update, frames=n_iters, fargs=(args,), interval=10
    )

    plt.show()
    anim.save()

    # start the animation!


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

    while j == (i := np.random.choice(range(N), p=p)):
        continue

    # assumes undirected graph
    if verbose:
        print(f'{j} -> {i}')
    graph.nodes[i][HEAT_ATTRIBUTE] += 1
    to_node = max(j, i)
    from_node = min(j, i)
    if to_node in graph[from_node]:
        graph[min(j, i)][max(j, i)][HEAT_ATTRIBUTE] += 1
    else:
        if verbose:
            print(f'There is no edge: {j} -> {i}')

    div = 0
    for l in range(N):
        div += A[l, j] * np.exp(alpha * p[l])

    for k in range(N):
        PI[k, j] = (A[k, j] * np.exp(alpha * p[k])) / div

    return i


# --------------------------------------------------------------

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

    edge_colors = nx.get_edge_attributes(graph, 'color').values()
    edge_weights = nx.get_edge_attributes(graph, 'weight').values()

    # this is roguhly the same thing as below but gives more control.
    # with this we can start the animation :)
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    pos = nx.spring_layout(graph)
    nodes = nx.draw_networkx_nodes(graph, pos)
    edges = nx.draw_networkx_edges(graph, pos)

    nodes.set_color(normalize_colors(node_heatmap))
    edges.set_color(normalize_colors(edge_colors))
    edges.set_linewidth(list(edge_weights))
    fig.show()

    pos = nx.spring_layout(graph)
    nx.draw(graph, pos,
            node_color=node_heatmap,
            edge_color=edge_colors,
            width=list(edge_weights))
    plt.show()
