from random_walks import random_walk_iteration
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

seed = 42
np.random.seed(seed)
font = {
    'weight': 'bold',
    'size': 16}

plt.rc('font', **font)


def show_time_to_convergence(
        graph: nx.Graph,
        graph_name: str,
        max_iters: int = 10000,
        alpha: int = 0.85):
    if not nx.is_connected(graph):
        raise Exception('Graph is disconnected')

    A = nx.adjacency_matrix(graph).toarray()
    N = len(A)

    PI = np.ones((N, N))
    PI /= np.sum(PI, axis=1)

    j = np.random.randint(N)
    visited = [False] * N
    visited[j] = True
    visited_counter = 1
    xs = [0]
    ys = [1]
    progress_bar = tqdm(total=N)
    for i in range(max_iters):
        j = random_walk_iteration(graph, A, PI, j, alpha=alpha, verbose=False)
        if not visited[j]:
            visited[j] = True
            visited_counter += 1
            progress_bar.update(1)

        xs.append(i + 1)
        ys.append(visited_counter)

        if visited_counter == N:
            break

    plt.figure(figsize=(10, 10))
    plt.plot(xs, ys, label='visited nodes', linewidth=5)
    plt.plot(xs, [N] * len(xs), label='total nodes', linewidth=3, linestyle='--')
    plt.xlabel('iterations')
    plt.ylabel('visited nodes')
    plt.title(f'{graph_name} graph: time to convergence, alpha={alpha}, nodes={N}')
    plt.show()
