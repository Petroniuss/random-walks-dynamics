from typing import Dict

import networkx as nx
import numpy as np
from tqdm import tqdm

from graphs import HEAT_ATTRIBUTE


class Simulation:
    def __init__(self, graph: nx.Graph,
                 PI: np.ndarray,
                 graph_name: str,
                 max_iters: int = 10000,
                 alpha: int = 0.85,
                 verbose: bool = False,
                 metrics: Dict[str, 'Metric'] = {},
                ):
        self.graph_name = graph_name
        self.max_iters = max_iters
        self.current_iter = 0
        self.verbose = verbose
        self.progress_bar = tqdm(total=self.max_iters)

        self.graph = graph
        self.PI = PI
        self.alpha = alpha
        self.A = nx.adjacency_matrix(graph).toarray()
        self.N = len(self.A)
        self.j = np.random.randint(self.N)

        # registering metrics must be the last step
        self.metrics = {metric_name: metric_constructor(self)
                        for metric_name, metric_constructor in metrics.items()}

    def run(self):
        for _ in self:
            pass
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_iter >= self.max_iters:
            raise StopIteration

        self.current_iter += 1
        self.progress_bar.update(1)

        self.__random_walk_step()

        for metric in self.metrics.values():
            metric.inc()

    def __random_walk_step(self):
        self.__update_PI()

        p = self.__get_p()

        if np.isnan(p).any():
            print(p.sum())
            print(p)

        while self.j == (i := np.random.choice(range(self.N), p=p)):
            continue

        if self.verbose:
            print(f'{self.j} -> {i}')

        # update heat
        self.graph.nodes[i][HEAT_ATTRIBUTE] += 1
        if i in self.graph[self.j]:
            self.graph[self.j][i][HEAT_ATTRIBUTE] += 1
        else:
            if self.verbose:
                print(f'There is no edge: {self.j} -> {i}')

        self.j = i

    def __get_p(self):
        return self.PI @ self.PI[:, self.j]

    def __update_PI(self):
        p = self.__get_p()
        div = 0
        for l in range(self.N):
            div += self.A[l, self.j] * np.exp(self.alpha * p[l])

        for k in range(self.N):
            self.PI[k, self.j] = (self.A[k, self.j] *
                                  np.exp(self.alpha * p[k])) / div

