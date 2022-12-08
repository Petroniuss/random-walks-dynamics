from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection, LineCollection
from tqdm import tqdm
from typing import List

HEAT_ATTRIBUTE = 'heat'
MATPLOTLIB_BACKEND = 'TkAgg'
SELECTED_NODE_COLOR = (0.0, 1.0, 0.0, 1.0)
PREV_SELECTED_NODE_COLOR = (0.0, 0.0, 1.0, 1.0)
CMAP = 'hot'

matplotlib.use(MATPLOTLIB_BACKEND)
plt.set_cmap(CMAP)


def dry_run_animation():
    """
    Assume undirected graph for now.
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
    selected: int
    edge_colors: List[tuple]
    node_colors: List[tuple]
    alpha: int
    verbose: bool
    progress_bar: tqdm


def animation_update(_frame: int, args: AnimationIterationArgs):
    """
    Called once per frame.
    """
    selected = random_walk_iteration(
        graph=args.graph,
        A=args.A,
        PI=args.PI,
        j=args.selected,
        alpha=args.alpha,
        verbose=args.verbose,
    )

    edge_weights = nx.get_edge_attributes(args.graph, HEAT_ATTRIBUTE).values()

    args.node_colors[args.selected] = normalize_color(args.graph.nodes[args.selected][HEAT_ATTRIBUTE])
    args.node_colors[selected] = SELECTED_NODE_COLOR

    # todo: check if it's possible to update only selected nodes/edges.
    args.drawn_nodes.set_color(args.node_colors)
    args.drawn_edges.set_color(args.edge_colors)
    args.drawn_edges.set_linewidth(list(edge_weights))
    args.progress_bar.update(1)
    args.selected = selected


def normalize_color(color) -> tuple:
    cmap = plt.get_cmap()
    vmin = 1
    vmax = 36
    color_normal = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap(color_normal(color))


def normalize_colors(colors):
    return [normalize_color(e) for e in colors]


def animate_random_walk(graph: nx.Graph,
                        alpha: int = 0.5,
                        n_iters: int = 100,
                        fps: int = 10,
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
    edge_colors = normalize_colors(nx.get_edge_attributes(graph, HEAT_ATTRIBUTE).values())
    node_colors = normalize_colors(nx.get_node_attributes(graph, HEAT_ATTRIBUTE).values())

    # move this to the algorithm.
    drawn_nodes.set_color(node_colors)
    drawn_edges.set_color(edge_colors)
    drawn_edges.set_linewidth(list(edge_weights))

    progress_bar = tqdm(total=n_iters)
    args = AnimationIterationArgs(
        graph=graph,
        drawn_nodes=drawn_nodes,
        drawn_edges=drawn_edges,
        A=A,
        PI=PI,
        selected=j,
        edge_colors=edge_colors,
        node_colors=node_colors,
        alpha=alpha,
        verbose=verbose,
        progress_bar=progress_bar
    )

    anim = FuncAnimation(
        fig, animation_update, frames=n_iters, fargs=(args,), interval=40, save_count=n_iters
    )

    anim.save('animation2.mp4', writer=animation.FFMpegWriter(fps=fps))
    # for interactive plotting.
    # plt.show()


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

    div = 0
    for l in range(N):
        div += A[l, j] * np.exp(alpha * p[l])

    for k in range(N):
        PI[k, j] = (A[k, j] * np.exp(alpha * p[k])) / div

    if verbose:
        print(f'{j} -> {i}')

    # update heat
    graph.nodes[i][HEAT_ATTRIBUTE] += 1
    if j in graph[i]:
        graph[j][i][HEAT_ATTRIBUTE] += 1
    else:
        # todo: is this a bug? I guess no..
        if verbose:
            print(f'There is no edge: {j} -> {i}')

    return i
