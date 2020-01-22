from graph import Graph, WrongAlphaValue
import matplotlib.pyplot as plt
import numpy as np
# import networkx as nx


def activate():
    plt.show()


def plot_comparision(g: Graph, delay_activate: bool = False):
    degree, degree_distribution = np.unique(g.vertex_degrees, return_counts=True)
    degree_distribution = degree_distribution / (g.m0 + g.tick)
    degree_distribution_theory = modified_ba_theoretical_curve(degree, g.alpha)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    plt.plot(degree, degree_distribution_theory, 'g', label='theoretical')

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])

    plt.title('Degree distribution (log-log scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')

    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    plt.plot(degree, degree_distribution_theory, 'g', label='theoretical')

    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])
    plt.title('Degree distribution (log-none scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')

    plt.legend()

    if not delay_activate:
        plt.show()


def plot_log_log(g: Graph, delay_activate: bool = False):
    degree, degree_distribution = np.unique(g.vertex_degrees, return_counts=True)
    degree_distribution = degree_distribution / (g.m0 + g.tick)
    degree_distribution_theory = modified_ba_theoretical_curve(degree, g.alpha)

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    if g.alpha > 0.999:
        a, y = best_fit(degree, degree_distribution, g.alpha)
        plt.plot(degree, y, 'b', label=f'best fit with slope = {a}')
        plt.plot(degree, degree_distribution_theory, 'g', label='theoretical with slope = -3')
    else:
        plt.plot(degree, degree_distribution_theory, 'g', label='theoretical')

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])

    plt.title('Degree distribution (log-log scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')

    plt.legend()

    if not delay_activate:
        plt.show()


def plot_log_none(g: Graph, delay_activate: bool = False):
    degree, degree_distribution = np.unique(g.vertex_degrees, return_counts=True)
    degree_distribution = degree_distribution / (g.m0 + g.tick)
    degree_distribution_theory = modified_ba_theoretical_curve(degree, g.alpha)

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    if g.alpha < 0.001:
        a, y = best_fit(degree, degree_distribution, g.alpha)
        plt.plot(degree, y, 'b', label=f'best fit with slope = {a}')
        plt.plot(degree, degree_distribution_theory, 'g', label='theoretical with slope = -1')
    else:
        plt.plot(degree, degree_distribution_theory, 'g', label='theoretical')

    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])

    plt.title('Degree distribution (log-none scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')

    plt.legend()

    if not delay_activate:
        plt.show()


def modified_ba_theoretical_curve(k: int, alpha: float, m: int = 1) -> float:
    """
    Functions that return value of probability that was calculated analytical using mean-field theory and stand for

    (m-2*(1-alpha)/alpha) ^ 2 * 2/alpha * (k + 2 - 2 alpha)^(-2-alpha), note that this equation will broke at alpha -> 0

    :param k: vertex degree
    :param alpha: alpha value
    :param m: at this moment always equal 1, originally this is yet another generalization of BA graph
    :return: probability of given vertex degree
    """
    if alpha > 0:
        m = 1
        ret = (m - 2 * (1 - alpha)/alpha)
        ret *= ret
        ret *= 2 / alpha
        ret *= (k + 2 - 2 * alpha)**(- 2 - alpha)
    else:
        ret = np.e / m * np.exp(- k / m)
    return ret


def best_fit(x: list, y: list, alpha: float) -> (float, list):
    if alpha > 0.999:
        x_true = np.log(x)
        y_true = np.log(y)
    elif alpha < 0.001:
        x_true = x
        y_true = np.log(y)
    else:
        raise WrongAlphaValue('in order to fit data, alpha must be 0 or 1')

    # calculate polynomial
    coeff = np.polyfit(x_true[:9*len(x_true)//10], y_true[:9*len(y_true)//10], 1)
    f = np.poly1d(coeff)

    # calculate new x's and y's
    y_ret: list = f(x_true)
    y_ret = np.exp(y_ret)
    return coeff[0], y_ret
