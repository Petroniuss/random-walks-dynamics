from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib import animation
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection, LineCollection
from tqdm import tqdm
from typing import List

HEAT_ATTRIBUTE = 'heat'
MATPLOTLIB_BACKEND = 'TkAgg'
SELECTED_NODE_COLOR = (0.0, 1.0, 0.0, 1.0)
CMAP = 'hot'

rcParams["figure.figsize"] = (16.4, 12.8)

# matplotlib.use(MATPLOTLIB_BACKEND)
plt.set_cmap(CMAP)


def dry_run_animation():
    """
    Assume undirected graph for now.
    """
    graph = nx.karate_club_graph()
    animate_random_walk(graph, animation_name='karate.mp4', n_iters=1000)


def dry_run_snap_graph():
    graph = nx.read_edgelist('./graphs/soc-sign-bitcoinalpha.csv',
                             delimiter=',',
                             nodetype=int,
                             data=[('rating', int), ('time', int)])
    print(f'Bitcoin graph: {nx.info(graph)}')
    graph = nx.relabel_nodes(graph, lambda e: e - 1)

    n = 250
    graph = nx.subgraph(graph, range(n))
    print(f'Bitcoin subgraph: {nx.info(graph)}')

    animation_name = f'bitcoin_nodes_{n}.mp4'
    animate_random_walk(graph, animation_name=animation_name, n_iters=1000, fps=20, node_size=50, edge_alpha=0.2)
    print(f'Generated {animation_name}')


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
                        animation_name: str = 'animation.mp4',
                        node_size: int = 300,
                        edge_alpha: float = 0.5,
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

    # for interactive plotting.
    # plt.show()
    anim.save(f'./animations/{animation_name}', writer=animation.FFMpegWriter(fps=fps))


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

    # todo: debug when this case occurs
    # my bet is that this happens when some node does not have any edges or something similar.
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
