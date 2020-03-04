from graph import Graph, WrongAlphaValue
import matplotlib.pyplot as plt
import numpy as np
import argparse


def parse_and_activate() -> None:
    """
    Function for command interface, which parse and choose right initial numbers/plots.
    """
    parser = argparse.ArgumentParser("Simulation of modified Barabasi-Albert graph and plot selected chart")
    parser.add_argument("m0", type=int,
                        help="size of the initial graph (at least 2)")
    parser.add_argument("alpha", type=float,
                        help="alpha value for modified BA model (from 0 to 1), "
                             "alpha = 1 mean pure BA, and alpha = 0 mean pure random")
    parser.add_argument("desired_size", type=int,
                        help="desired size of the graph at the end")
    parser.add_argument("-c", "--compare", action="store_true",
                        help="compare graph on two chart - log-log scale and log-none scale")
    parser.add_argument("-ll", "--loglog", action="store_true",
                        help="plot on log-log scale and perform fit if alpha = 1")
    parser.add_argument("-ln", "--lognone", action="store_true",
                        help="plot on log-none scale and perform fit if alpha = 0")
    args = parser.parse_args()

    print(f'Simulate graph with initial size m0={args.m0}, alpha value={args.alpha}, to size {args.desired_size}.')
    g = Graph(m0=args.m0, alpha=args.alpha)
    if args.compare or args.loglog or args.lognone:
        g.update_graph_to_size(args.desired_size)
    else:
        g.test_speed(args.desired_size)

    if args.compare:
        plot_comparision(g)
    if args.loglog:
        plot_log_log(g)
    if args.lognone:
        plot_log_none(g)
    plt.show()


def plot_comparision(g: Graph) -> None:
    """
    Create and display two chart - one with log-log scale and second with log-none scale of simulation data and
    theoretical curve.

    Function that first compute degree_distribution just by adding how many given vertex degree appeared in list. Then
    it is normalized and finally theoretical curve is computed. Note that mentioned curve is calculated using mean-field
    theory, and can be different from simulation data, especially when alpha is closer to 0 than to 1 - I am still
    working about that - possibly it is because it reaches its limit at infinity much slower compared to alpha=1.
    """
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


def plot_log_log(g: Graph) -> None:
    """
    Create and display plot with log-log scale of simulation data and theoretical curve. Additionally if alpha
    is close to 1, it will fit data to linear function.

    Function that first compute degree_distribution just by adding how many given vertex degree appeared in list. Then
    it is normalized and finally theoretical curve is computed. Note that mentioned curve is calculated using mean-field
    theory, and can be different from simulation data, especially when alpha is closer to 0 than to 1 - I am still
    working about that - possibly it is because it reaches its limit at infinity much slower compared to alpha=1.
    """
    degree, degree_distribution = np.unique(g.vertex_degrees, return_counts=True)
    degree_distribution = degree_distribution / (g.m0 + g.tick)
    degree_distribution_theory = modified_ba_theoretical_curve(degree, g.alpha)

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    if g.alpha > 0.999:
        a, degree_fit, degree_distribution_fit = best_fit(degree, degree_distribution, g.alpha)
        plt.plot(degree_fit, degree_distribution_fit, 'b', label=f'best fit with slope = {a}')
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


def plot_log_none(g: Graph) -> None:
    """
    Create and display plot with log-none scale of simulation data and theoretical curve. Additionally if alpha
    is close to 0, it will fit data to linear function.

    Function that first compute degree_distribution just by adding how many given vertex degree appeared in list. Then
    it is normalized and finally theoretical curve is computed. Note that mentioned curve is calculated using mean-field
    theory, and can be different from simulation data, especially when alpha is closer to 0 than to 1 - I am still
    working about that - possibly it is because it reaches its limit at infinity much slower compared to alpha=1.
    """
    degree, degree_distribution = np.unique(g.vertex_degrees, return_counts=True)
    degree_distribution = degree_distribution / (g.m0 + g.tick)
    degree_distribution_theory = modified_ba_theoretical_curve(degree, g.alpha)

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    if g.alpha < 0.001:
        a, degree_fit, degree_distribution_fit = best_fit(degree, degree_distribution, g.alpha)
        plt.plot(degree_fit, degree_distribution_fit, 'b', label=f'best fit with slope = {a}')
        plt.plot(degree, degree_distribution_theory, 'g', label=f'theoretical with slope = {-np.log(2)}')
    else:
        plt.plot(degree, degree_distribution_theory, 'g', label='theoretical')

    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])

    plt.title('Degree distribution (log-none scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')

    plt.legend()


def modified_ba_theoretical_curve(k: int, alpha: float, m: int = 1) -> float:
    """
    Functions that return value of probability that was calculated analytical using mean-field theory and stand for
    if alpha!=0

    (m-2*(1-alpha)/alpha) ^ 2 * 2/alpha * (k + 2 - 2 alpha)^(-2-alpha), note that this equation will broke at alpha -> 0

    if alpha=0 :
    e/m * exp(-k/m)

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
        ret = m ** (k - m) / (m+1) ** (k - m + 1)
        # ret = np.e / m * np.exp(- k / m)
    return ret


def best_fit(x: list, y: list, alpha: float) -> (float, list, list):
    """
    Function that can be used only when alpha is close to 0 or 1 - then we know how to linear theoretical curve and
    use least square method, which is actually used there for both cases

    :param x: list of degrees
    :param y: list of degrees distribution
    :param alpha:
    :return: predicted alpha, and x (degree list) alongside with predicted y (degrees distribution list)
    """
    if alpha > 0.999:
        x_true = np.log(x)
        y_true = np.log(y)
        # calculate polynomial from position 'start' to position 'end' - in original BA model we want to approach only
        # "middle" points
        start = 1 + 2 * len(x_true) // 100
        end = 50 * len(x_true) // 100
    elif alpha < 0.001:
        x_true = x
        y_true = np.log(y)
        # calculate polynomial from position 'start' to position 'end'
        start = 0
        end = len(x_true)
    else:
        raise WrongAlphaValue('in order to fit data, alpha must be 0 or 1')

    coeff = np.polyfit(x_true[start:end], y_true[start:end], 1)
    f = np.poly1d(coeff)

    # calculate new x's and y's
    y_ret: list = f(x_true)
    if alpha > 0.999:
        y_ret = np.exp(y_ret)
        x_ret = np.exp(x_true)
    elif alpha < 0.001:
        y_ret = np.exp(y_ret)
        x_ret = x_true
    else:
        raise WrongAlphaValue('in order to fit data, alpha must be 0 or 1')
    return coeff[0], x_ret[start:end], y_ret[start:end]
