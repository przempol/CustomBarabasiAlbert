import numpy as np
import time


class Graph:
    """
    Class used to implement modified Barbasi Albert Graph.


    """
    def __init__(self, m0: int = 3, alpha: float = 1):
        """
        Create instance of Barabasi-Albert Graph with given starting size 'm0', and custom parameter 'alpha'
        what indices probability of using PAR (preferential attachment rule - see Barabasi-Albert graphs) over random
        choose of nodes.

        :param m0: size of graph in time = 0, default 3
        :param alpha: float from 0 do 1, describing probability of using preferential attachment rule,
        if 1 => will only use PAR (standard BA graph), if 0 => never use PAR (random graphs)

        """
        self.__m0: int = m0
        self.__alpha: float = alpha
        self.__tick: int = 0
        self.__nodes: list = np.concatenate([np.full(m0 - 1, ii) for ii in range(m0)]).tolist()

    @property
    def m0(self):
        return self.__m0

    @property
    def alpha(self):
        return self.__alpha

    @property
    def tick(self):
        return self.__tick

    @property
    def nodes(self):
        return self.__nodes

    @tick.setter
    def tick(self, tick):
        if not isinstance(tick, int):
            print("Not number value!")
        else:
            self.__tick = tick

    def update(self, desired_size: int):
        """
        Update graph to desired size (number of nodes), what must obviously must be larger than current size
        While updating, it is only using graphs.append() function

        :param desired_size: integer (that must be larger than actual size of graph) of desired size
        """
        repeats = desired_size - self.m0 - self.tick
        for i in range(repeats):
            self.append()

    def append(self):
        """
        Append just one node to the current graph. While it is modified Barabasi-Albert graphs, it compare 'alpha' value
        with random float number from 0 to 1 generated from uniform distribution. If 'alpha' is larger than random
        number, it will choose node using preferential attachment rule (PAR), in other case will choose random node.
        """
        # x is current number of adding node
        x = self.tick + self.m0
        self.__nodes.append(x)
        rnd = np.random.rand()
        if self.alpha > rnd:
            size = self.m0 * (self.m0 - 1) + 2 * self.tick
            # y is imitating preferential attachment rule
            y = self.__nodes[np.random.randint(size)]
        else:
            # y is imitating random choosing
            y = np.random.randint(x)

        self.__nodes.append(y)
        self.tick += 1

    def test_speed(self, desired_size) -> float:
        """
        Function that compute and return time (in seconds) needed to create graph of desired size, with initial values
        the same as current graph. For comparision, using matrix methods implemented in mathematica, it took about
        30 minutes to make graph of 1e4 nodes, while using list methods implemented in python, it took about 1 second
        to make graph of 1e5 nodes

        :param desired_size: integer (that must be larger than actual size of graph) of desired size
        :return time:
        """
        start = time.time()

        tmp_graph = Graph(self.m0, self.alpha)
        repeats = desired_size - tmp_graph.m0 - tmp_graph.tick

        for i in range(repeats):
            tmp_graph.append()

        end = time.time()
        print(f'Creating modified BA graph of size {desired_size} lasted {end - start} second')
        return end - start
