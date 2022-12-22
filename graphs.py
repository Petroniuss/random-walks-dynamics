import networkx as nx

HEAT_ATTRIBUTE = 'heat'


def decorate_graph_with_attributes(graph: nx.Graph):
    nx.set_edge_attributes(graph, 1, HEAT_ATTRIBUTE)
    nx.set_node_attributes(graph, 1, HEAT_ATTRIBUTE)

    return graph


def graph_karate_club():
    return decorate_graph_with_attributes(nx.karate_club_graph())


def select_max_connected_component(graph: nx.Graph):
    connected_nodes = max(nx.connected_components(graph), key=len)
    graph = nx.subgraph(graph, connected_nodes).copy()
    graph = nx.convert_node_labels_to_integers(graph)

    return graph


def graph_bitcoin(n: int = 100):
    graph = nx.read_edgelist('./graphs/soc-sign-bitcoinalpha.csv',
                             delimiter=',',
                             nodetype=int,
                             data=[('rating', int), ('time', int)])
    print(f'Bitcoin graph: {nx.info(graph)}')
    graph = nx.subgraph(graph, range(1, n))
    subgraph = select_max_connected_component(graph)
    print(f'Subgraph: {nx.info(subgraph)}')

    return decorate_graph_with_attributes(subgraph)


def graph_bridge(n: int = 250):
    graph = nx.read_edgelist('./graphs/soc-sign-bitcoinalpha.csv',
                             delimiter=',',
                             nodetype=int,
                             data=[('rating', int), ('time', int)])
    print(f'Bitcoin graph: {nx.info(graph)}')

    subgraph_1 = nx.subgraph(graph, range(1, n))
    subgraph_1 = select_max_connected_component(subgraph_1)

    subgraph_2 = nx.subgraph(graph, range(n, 2 * n))
    subgraph_2 = select_max_connected_component(subgraph_2)

    # connect two subgraphs
    max_node = max(subgraph_1.nodes)
    subgraph_2 = nx.convert_node_labels_to_integers(subgraph_2, first_label=max_node + 1)
    graph = nx.compose(subgraph_1, subgraph_2)
    graph.add_edge(max_node, max_node + 1)

    print(f'Selected subgraph: {nx.info(graph)}')
    return decorate_graph_with_attributes(graph)
