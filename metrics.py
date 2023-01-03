from abc import ABC, abstractmethod
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from simulation import Simulation

seed = 42
np.random.seed(seed)
font = {
    'weight': 'bold',
    'size': 16
}

plt.rc('font', **font)


class Metric(ABC):
    def __init__(self, sim: Simulation):
        self.sim = sim

    @abstractmethod
    def inc(self):
        pass


class ConvergenceTime(Metric):
    def __init__(self, sim):
        self.sim = sim
        self.visited = [False] * self.sim.N
        self.visited[self.sim.j] = True
        self.visited_counter = 1
        self.xs = [0]
        self.ys = [1]

    def inc(self):
        if self.visited_counter == self.sim.N:
            return

        if not self.visited[self.sim.j]:
            self.visited[self.sim.j] = True
            self.visited_counter += 1

        self.xs.append(self.sim.current_iter)
        self.ys.append(self.visited_counter)

    def show(convergence_time: 'ConvergenceTime'):
        # plt.figure(figsize=(10, 10))
        plt.plot(convergence_time.xs, convergence_time.ys,
                 label='visited nodes', linewidth=5)
        plt.plot(convergence_time.xs, [convergence_time.sim.N] * len(convergence_time.xs), label='total nodes',
                 linewidth=3, linestyle='--')
        plt.xlabel('iterations')
        plt.ylabel('visited nodes')
        plt.title(
            f'{convergence_time.sim.graph_name} graph: time to convergence, alpha={convergence_time.sim.alpha}, nodes={convergence_time.sim.N}')
        plt.legend()
        plt.show()


class NodeVisits(Metric):
    def __init__(self, sim):
        self.sim = sim
        self.heatmap = defaultdict(lambda: 0)

    def inc(self):
        self.heatmap[self.sim.j] += 1
