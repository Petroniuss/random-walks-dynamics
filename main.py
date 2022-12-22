from random_walks import dry_run_snap_graph
from statistics import show_time_to_convergence
from graphs import graph_bitcoin, graph_karate_club, graph_bridge

# 1. Extract graph definition from core algorithm.
# 2. Create animation:
#   Animations using networkxx
#   https://colab.research.google.com/drive/1ed0fAC6rUMcScJYxv0Kc0Ox-wDBf2FSR?usp=sharing#scrollTo=jgtW7fk5JFRD
# 3. Take a bigger graph! Stanford graph collection.
# 4. Go back to paper and see what else is there :)


if __name__ == '__main__':
    # show_time_to_convergence(graph_karate_club(), graph_name='karate')
    # show_time_to_convergence(graph_bitcoin(500), graph_name='bitcoin')
    # show_time_to_convergence(graph_bitcoin(500), graph_name='bitcoin', alpha=7.0)
    # show_time_to_convergence(graph_bitcoin(500), graph_name='bitcoin', alpha=0.1)
    graph = graph_bridge(250)
    show_time_to_convergence(graph_bridge(250), graph_name='bridge', alpha=0.1)
    # show_time_to_convergence(graph_bitcoin(n=100))
    # dry_run_snap_graph()
