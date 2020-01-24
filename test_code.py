import pytest

from graph import (
    Graph,
    WrongInitialSize,
    WrongAlphaValue,
    WrongGraphSize
)


@pytest.fixture
def graph(m0, alpha):
    return Graph(m0=m0, alpha=alpha)


@pytest.mark.parametrize("m0, alpha, expected_nodes, expected_vertex_degrees",
                         [(2, 1, [0, 1], [1, 1]),
                          (3, 1, [0, 0, 1, 1, 2, 2], [2, 2, 2])
                          ])
def test_valid_initial_data(m0, alpha, expected_nodes, expected_vertex_degrees, graph):
    assert graph.nodes == expected_nodes
    assert graph.vertex_degrees == expected_vertex_degrees


@pytest.mark.parametrize("m0, alpha, desired_size, expected_size",
                         [(2, 1, 50, 50),
                          (2, 0.5, 50, 50),
                          (49, 0, 50, 50),
                          ])
def test_valid_size(m0, alpha, desired_size, expected_size, graph):
    graph.update_graph_to_size(desired_size)
    assert graph.size == expected_size


@pytest.mark.parametrize("m0, alpha, expected_degree_before",
                         [(2, 1, [1, 1]),
                          (3, 1, [2, 2, 2]),
                          (4, 1, [3, 3, 3, 3])
                          ])
def test_valid_vertex_degrees(m0, alpha, expected_degree_before, graph):
    assert graph.vertex_degrees == expected_degree_before


@pytest.mark.parametrize("m0, alpha, desired_size, expected",
                         [(1, 1, 5, WrongInitialSize),
                          ('2', 1, 5, WrongInitialSize),
                          (2., 1, 5, WrongInitialSize),
                          (-1, 1, 5, WrongInitialSize),
                          (2, 1.01, 5, WrongAlphaValue),
                          (2, -1, 5, WrongAlphaValue),
                          (2, -0.1, 5, WrongAlphaValue),
                          (10, 0, 1e5, WrongGraphSize),
                          (10, 0, 5, WrongGraphSize),
                          (10, 0, 100., WrongGraphSize)
                          ])
def test_run_invalid_data(m0, alpha, desired_size, expected):
    with pytest.raises(expected):
        g = Graph(m0=m0, alpha=alpha)
        g.update_graph_to_size(desired_size)
