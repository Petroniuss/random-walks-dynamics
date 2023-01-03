from dataclasses import dataclass
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import animation, rcParams
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PathCollection

import pi
from graphs import HEAT_ATTRIBUTE, graph_bitcoin, graph_karate_club
from simulation import Simulation

# MATPLOTLIB_BACKEND = 'TkAgg'
SELECTED_NODE_COLOR = (0.0, 1.0, 0.0, 1.0)
CMAP = 'hot'

rcParams["figure.figsize"] = (16.4, 12.8)

plt.set_cmap(CMAP)


def dry_run_animation():
    """
    Assume undirected graph for now.
    """
    graph = graph_karate_club()
    PI = pi.uniform()
    animate_random_walk(graph, animation_name='karate.mp4', n_iters=1000)


def dry_run_snap_graph(n: int = 250):
    graph = graph_bitcoin(n)
    print(f'Bitcoin subgraph: {nx.info(graph)}')
    animation_name = f'bitcoin_nodes_{n}.mp4'
    animate_random_walk(graph, animation_name=animation_name, n_iters=1000, fps=20, node_size=50, edge_alpha=0.2)
    print(f'Generated {animation_name}')


@dataclass
class AnimationIterationArgs:
    sim: Simulation
    drawn_nodes: PathCollection
    drawn_edges: LineCollection
    edge_colors: List[tuple]
    node_colors: List[tuple]

def animation_update(_frame: int, args: AnimationIterationArgs):
    """
    Called once per frame.
    """
    old_selected = args.sim.j
    next(args.sim)
    new_selected = args.sim.j

    edge_weights = nx.get_edge_attributes(args.sim.graph, HEAT_ATTRIBUTE).values()

    args.node_colors[old_selected] = normalize_color(args.sim.graph.nodes[old_selected][HEAT_ATTRIBUTE])
    args.node_colors[new_selected] = SELECTED_NODE_COLOR

    # todo: check if it's possible to update only selected nodes/edges.
    args.drawn_nodes.set_color(args.node_colors)
    args.drawn_edges.set_color(args.edge_colors)
    args.drawn_edges.set_linewidth(list(edge_weights))


def normalize_color(color) -> tuple:
    cmap = plt.get_cmap()
    vmin = 1
    vmax = 36
    color_normal = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap(color_normal(color))


def normalize_colors(colors):
    return [normalize_color(e) for e in colors]


def animate_random_walk(sim: Simulation,
                        animation_name: str = 'animation.mp4',
                        node_size: int = 300,
                        edge_alpha: float = 0.5,
                        fps: int = 10,
                    ):
    # matplotlib.use(MATPLOTLIB_BACKEND)

    # animation initial state
    fig = plt.figure()
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    pos = nx.spring_layout(sim.graph)
    drawn_nodes = nx.draw_networkx_nodes(sim.graph, pos)
    drawn_edges = nx.draw_networkx_edges(sim.graph, pos)

    nx.set_edge_attributes(sim.graph, 1, HEAT_ATTRIBUTE)
    nx.set_node_attributes(sim.graph, 1, HEAT_ATTRIBUTE)

    edge_weights = nx.get_edge_attributes(sim.graph, HEAT_ATTRIBUTE).values()
    edge_colors = normalize_colors(nx.get_edge_attributes(sim.graph, HEAT_ATTRIBUTE).values())
    node_colors = normalize_colors(nx.get_node_attributes(sim.graph, HEAT_ATTRIBUTE).values())

    drawn_nodes.set_color(node_colors)
    drawn_edges.set_color(edge_colors)
    drawn_edges.set_linewidth(list(edge_weights))

    args = AnimationIterationArgs(
        sim=sim,
        drawn_nodes=drawn_nodes,
        drawn_edges=drawn_edges,
        edge_colors=edge_colors,
        node_colors=node_colors,
    )

    anim = FuncAnimation(
        fig, animation_update, frames=sim.max_iters - 1, fargs=(args,), interval=40, save_count=sim.max_iters - 1
    )

    # for interactive plotting.
    # plt.show()
    anim.save(f'./animations/{animation_name}', writer=animation.FFMpegWriter(fps=fps))
