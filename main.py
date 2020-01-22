from graph import Graph
from gui import chart

if __name__ == "__main__":
    g = Graph(m0=2, alpha=1)
    g.update_graph_to_size(int(1.6e5))
    g.single_update()
    desired_size = int(1.6e5)
    time = g.test_speed(desired_size)
    print(f'Creating modified BA graph of size {desired_size} lasted {time} second')

    # print(f'done {g.alpha}, zas rozmiar to {g.size}')
    # print(g.vertex_degrees)
    # chart(g)


