from graph import Graph
import matplotlib.pyplot as plt
import math
import numpy as np
# import networkx as nx


def chart(g: Graph):
    unique, degree_distribution = np.unique(g.nodes, return_counts=True)
    # print(f'obecna wartosc ticku to {g.tick}, zas rozmiar klastra to {len(unique)}')
    unique, degree_distribution = np.unique(degree_distribution, return_counts=True)
    degree_distribution = degree_distribution / (g.m0 + g.tick)

    x_theory = [unique[0], math.sqrt(unique[-1])]
    y_theory = [1, (x_theory[-1] ** (-3))]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(unique, degree_distribution, 'r.')
    plt.plot(x_theory, y_theory, 'g')

    plt.xscale("log")
    plt.yscale("log")

    plt.title('Degree distribution (log-log scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')

    plt.subplot(1, 2, 2)
    plt.plot(unique, degree_distribution, 'r.')

    plt.yscale("log")

    plt.title('Degree distribution (log-none scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')

    # plt.annotate('$P(k) \propto ~ k^{-3}$', xy=(10, 10 ** (-3)), xytext=(1, 10 ** (-3)),
    #              arrowprops=dict(facecolor='black', shrink=0.05),
    #              )

    # plt.figure()
    # plt.plot(unique, degree_distribution, 'r.')

    plt.show()
