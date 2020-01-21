from graph import Graph
from gui import chart

if __name__ == "__main__":
    x = Graph(m0=3, alpha=0.)
    x.update(int(2e5))

    x.test_speed(int(1e5))
    chart(x)


