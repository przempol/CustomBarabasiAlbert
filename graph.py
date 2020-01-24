import numpy as np
import time


class GraphError(Exception):
    pass


class WrongInitialValue(GraphError):
    pass


class WrongAlphaValue(GraphError):
    pass


class WrongGraphSize(GraphError):
    pass


class Graph:
    """
    Class used to implement modified Barbasi Albert Graph. It contains values:

    int m0 - initial size of complete graph

    float alpha - float from 0 to 1, in particular if 1 it is equal to standard Barbasi Albert Graph,
    if 0 then it is random graph

    int tick - current time

    list nodes - list of where are nodes currently connected - it is a much faster way to store state of graph,
    than the matrix of adjacency

    list vertex_degrees - list of currently vertex degrees

    int size - size of graph (number of vertexes)


    And following methods are implemented

    Graph.single_update() - update state of Graph by adding one node

    Graph.update_graph_to_size(size) - update state of Graph by to desired size

    Graph.test_speed(size) - measure time to create graph of desired size, without changing current state
    """
    def __init__(self, m0: int = 3, alpha: float = 1.):
        """
        Create instance of Barabasi-Albert Graph with given starting size 'm0', and custom parameter 'alpha'
        what indices probability of using PAR (preferential attachment rule - see Barabasi-Albert graphs) over random
        choose of nodes.

        :param m0: size of graph in time = 0, default 3
        :param alpha: float from 0 do 1, describing probability of using preferential attachment rule,
        if 1 => will only use PAR (standard BA graph), if 0 => never use PAR (random graphs)

        """
        self.m0 = m0
        self.alpha = alpha
        self.__tick: int = 0
        self.__nodes: list = list(np.concatenate([np.full(m0 - 1, ii) for ii in range(m0)]).tolist())

    @property
    def m0(self) -> int:
        return self.__m0

    @m0.setter
    def m0(self, m0) -> None:
        if isinstance(m0, int):
            if m0 > 1:
                self.__m0: int = m0
            else:
                raise WrongInitialValue('m0 must be integer larger than 1')
        else:
            raise WrongInitialValue('m0 must be integer larger than 1')

    @property
    def alpha(self) -> float:
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        if isinstance(alpha, (float, int)):
            if 0 <= alpha <= 1:
                self.__alpha: float = float(alpha)
            else:
                raise WrongAlphaValue('alpha must be a number from 0 to 1')
        else:
            raise WrongAlphaValue('alpha must be a number from 0 to 1')

    @property
    def tick(self) -> int:
        return self.__tick

    @property
    def nodes(self) -> list:
        return self.__nodes

    @property
    def vertex_degrees(self) -> list:
        unique, degree_distribution = np.unique(self.nodes, return_counts=True)
        return degree_distribution.tolist()

    @property
    def size(self) -> int:
        return self.m0 + self.tick

    def update_graph_to_size(self, desired_size: int):
        """
        Update graph to desired size (number of nodes), what must obviously must be larger than current size
        While updating, it is only using graphs.append() function

        :param desired_size: integer (that must be larger than actual size of graph) of desired size
        """
        if isinstance(desired_size, int):
            repeats = desired_size - self.size
            if repeats <= 0:
                raise WrongGraphSize('desired size of graph must be integer larger than current size')
        else:
            raise WrongGraphSize('desired size of graph must be integer larger than current size')

        for i in range(repeats):
            self.single_update()

    def single_update(self):
        """
        Append just one node to the current graph. While it is modified Barabasi-Albert graphs, it compare 'alpha' value
        with random float number from 0 to 1 generated from uniform distribution. If 'alpha' is larger than random
        number, it will choose node using preferential attachment rule (PAR), in other case will choose random node.
        """
        # x is current number of adding node
        x = self.tick + self.m0
        self.__nodes.append(x)
        # rnd is number from 0 to 1, so if alpha = 1, then its always bigger than rnd
        rnd = np.random.rand()
        if self.alpha > rnd:
            size = self.m0 * (self.m0 - 1) + 2 * self.tick
            # y is imitating preferential attachment rule
            y = self.__nodes[np.random.randint(size)]
        else:
            # y is imitating random choosing
            y = np.random.randint(x)

        self.__nodes.append(y)
        self.__tick += 1

    def test_speed(self, desired_size: int) -> float:
        """
        Function that measure and return time (in seconds) needed to create graph of desired size, with initial values
        the same as current graph. For comparision, using matrix methods implemented in mathematica, it took about
        30 minutes to make graph of 1e4 nodes, while using list methods implemented in python, it took about 1 second
        to make graph of 1e5 nodes.

        :param desired_size: integer (that must be larger than actual size of graph) of desired size
        :return: time
        """
        if isinstance(desired_size, int):
            if desired_size <= 0:
                raise WrongGraphSize
        else:
            raise WrongGraphSize

        start = time.time()
        tmp_graph = Graph(self.m0, self.alpha)

        repeats = desired_size - tmp_graph.size
        for i in range(repeats):
            tmp_graph.single_update()

        end = time.time()
        print(f'It took {end - start} seconds to simulate whole graph.')
        return end - start
