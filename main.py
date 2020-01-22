from graph import Graph
from gui import plot_comparision, plot_log_none, plot_log_log, activate

if __name__ == "__main__":
    g = Graph(m0=2, alpha=1)
    g.update_graph_to_size(20)
    g.update_graph_to_size(int(1e6))
    g.single_update()

    plot_comparision(g, True)
    plot_log_log(g, True)
    plot_log_none(g, True)
    activate()

