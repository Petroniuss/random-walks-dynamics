import matplotlib.pyplot as plt

import pi
from animation import (animate_random_walk, dry_run_animation,
                       dry_run_snap_graph)
from graphs import graph_bitcoin, graph_bridge, graph_karate_club
from metrics import ConvergenceTime
from simulation import Simulation

# 1. Extract graph definition from core algorithm.
# 2. Create animation:
#   Animations using networkxx
#   https://colab.research.google.com/drive/1ed0fAC6rUMcScJYxv0Kc0Ox-wDBf2FSR?usp=sharing#scrollTo=jgtW7fk5JFRD
# 3. Take a bigger graph! Stanford graph collection.
# 4. Go back to paper and see what else is there :)


if __name__ == '__main__':
    karate = graph_karate_club()

    karate_sim = Simulation(
        karate,
        pi.uniform(karate),
        max_iters=100,
        graph_name='karate',
        metrics={'ConvergenceTime': ConvergenceTime},
    )

    animation_name = f'karate.mp4'
    animate_random_walk(karate_sim, animation_name=animation_name, fps=20, node_size=50, edge_alpha=0.2)
    print(f'Generated {animation_name}')

    plt.close()

    ConvergenceTime.show(karate_sim.metrics['ConvergenceTime'])
