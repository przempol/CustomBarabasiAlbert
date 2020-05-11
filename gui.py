from graph import Graph
from fitters import *
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
                        help="plot on log-log scale and perform fit")
    parser.add_argument("-ln", "--lognone", action="store_true",
                        help="plot on log-none scale and perform fit")
    parser.add_argument("-fit", "--fit", action="store_true",
                        help="Perform analytical fit using maximum likelihood method")
    parser.add_argument("-m1", "--method1", action="store_true",
                        help="plot with normal scale")
    parser.add_argument("-m2", "--method2", action="store_true",
                        help="plot with log-log scale")
    parser.add_argument("-m3", "--method3", action="store_true",
                        help="plot with log-log scale and logarithmic bins")
    parser.add_argument("-m4", "--method4", action="store_true",
                        help="plot with log-log scale, but cumulative distribution")
    args = parser.parse_args()

    print(f'Simulate graph with initial size m0={args.m0}, alpha value={args.alpha}, to size {args.desired_size}.')
    g = Graph(m0=args.m0, alpha=args.alpha)
    if args.compare or args.loglog or args.lognone or args.method1 or args.method2 or args.method3 or args.method4 or \
            args.fit:
        g.update_graph_to_size(args.desired_size)
    else:
        g.test_speed(args.desired_size)

    if args.compare:
        plot_comparision(g.vertex_degree_distribution, g.alpha)
    if args.loglog:
        plot_log_log(g.vertex_degree_distribution, g.alpha)
    if args.lognone:
        plot_log_none(g.vertex_degree_distribution, g.alpha)
    if args.method1:
        plot_method1(g.vertex_degree_distribution)
    if args.method2:
        plot_method2(g.vertex_degree_distribution)
    if args.method3:
        plot_method3(g.vertex_degree_distribution)
    if args.method4:
        plot_method4(g.vertex_degree_distribution)
    if args.fit:
        for ii in range(20):
            mlm_fit(g.vertex_degrees, ii*10+10)

    plt.show()


def plot_comparision(vertex_degree_distribution: (list, list), alpha: float) -> None:
    """
    Create and display two chart - one with log-log scale and second with log-none scale of simulation data and
    theoretical curve.

    Function that first compute degree_distribution just by adding how many given vertex degree appeared in list. Then
    it is normalized and finally theoretical curve is computed. Note that mentioned curve is calculated using mean-field
    theory, and can be different from simulation data, especially when alpha is closer to 0 than to 1 - I am still
    working about that - possibly it is because it reaches its limit at infinity much slower compared to alpha=1.
    """
    degree, degree_distribution = vertex_degree_distribution
    degree_distribution_mft = solution_mft(degree, alpha)
    degree_distribution_me = solution_me(degree, alpha)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    plt.plot(degree, degree_distribution_mft, 'g', label='Analytical solution (MFT)')
    plt.plot(degree, degree_distribution_me, 'b', label='Analytical solution (ME)')

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])
    plt.title('Degree distribution (log-log scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    plt.plot(degree, degree_distribution_mft, 'g', label='Analytical solution (MFT)')
    plt.plot(degree, degree_distribution_me, 'b', label='Analytical solution (ME)')

    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])
    plt.title('Degree distribution (log-none scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')
    plt.legend()


def plot_log_log(vertex_degree_distribution: (list, list), alpha: float) -> None:
    """
    Create and display plot with log-log scale of simulation data and theoretical curve. Additionally it will fit
    data to linear function.

    Note: Note that mentioned curve is calculated by using mean-field theory, and can be different from simulation data,
    especially when alpha is closer to 0 than to 1 - I am still working about that - possibly it is because it reaches
    its limit at infinity much slower compared to alpha=1.

    Note2: using Master Equation we obtain power law, with slope in [3, infinity). But if alpha is closer to 0
    (slope -> infinity) then we observe power law for larger degree !!! Thus we must use larger data, like 10 million...
    """
    degree, degree_distribution = vertex_degree_distribution
    slope, sigma, degree, degree_distribution_ls = ls_fit(degree, degree_distribution)
    degree_distribution_mft = solution_mft(degree, alpha)
    degree_distribution_me = solution_me(degree, alpha)

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    plt.plot(degree, degree_distribution_ls, 'g', label=f'best fit with slope = {slope:.3f} +- {sigma:.3f}')
    plt.plot(degree, degree_distribution_mft, 'b', label=f'theoretical curve (MFT)')
    plt.plot(degree, degree_distribution_me, 'violet', label=f'theoretical curve (ME)')
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])

    plt.title('Degree distribution (log-log scale)')
    plt.xlabel('Node degree')
    plt.ylabel('Probability')
    plt.legend()


def plot_log_none(vertex_degree_distribution: (list, list), alpha: float) -> None:
    """
    Create and display plot with log-none scale of simulation data. Additionally if alpha
    is close to 0, it will plot theoretical curve using MFT and ME.

    Function that plot data, and if alpha=~0 - theoretical curve is computed and plotter. Note that mentioned curve
    is calculated using mean-field theory(MFT) and Master equation(ME), thus we will see difference between them.
    Spoiler alert: ME is much better than MFT. It is because MFT is right, only when m0->infty, but ME works every time
    """
    degree, degree_distribution = vertex_degree_distribution

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')

    # I am doing np.exp(degree) only because ls_fit() automatically log data, and in this scenario we don't want to
    slope, sigma, degree, degree_distribution_ls = ls_fit(np.exp(degree), degree_distribution)
    # and we'are fixing once again output from ls_fit()
    degree = np.log(degree)
    plt.plot(degree, degree_distribution_ls, 'g', label=f'best fit with slope = {slope:.3f} +- {sigma:.3f}')

    if alpha < 0.001:
        degree_distribution_mft = solution_mft(degree, alpha)
        degree_distribution_me = solution_me(degree, alpha)
        plt.plot(degree, degree_distribution_mft, 'b', label=f'theoretical curve (MFT) = {-1}')
        plt.plot(degree, degree_distribution_me, 'violet', label=f'theoretical curve (ME) = {-np.log(2):.3}')

    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])

    plt.title('Degree distribution (log-none scale)')
    plt.xlabel('Node degree')
    plt.ylabel('probability')
    plt.legend()


def plot_method1(vertex_degree_distribution: (list, list)) -> None:
    """
    Create and display "normal" plot. This is first part from 4 methods what I want to compare in my thesis:
    1. normal - non log scales
    2. log-log - log scales
    3. log-bin - log scales & logarithmic bins
    4. log-log with cumulative adding
    """
    degree, degree_distribution = vertex_degree_distribution

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')

    plt.title('Degree distribution (log-none scale)')
    plt.xlabel('Node degree')
    plt.ylabel('Probability')
    plt.legend()


def plot_method2(vertex_degree_distribution: (list, list)) -> None:
    """
    Create and display log-log plot. This is second part from 4 methods what I want to compare in my thesis:
    1. normal - non log scales
    2. log-log - log scales
    3. log-bin - log scales & logarithmic bins
    4. log-log with cumulative adding
    """
    degree, degree_distribution = vertex_degree_distribution
    slope, sigma, degree, degree_distribution_ls = ls_fit(degree, degree_distribution)

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    plt.plot(degree, degree_distribution_ls, 'b', label=f'best fit with slope = {slope:.3f} +- {sigma:.3f}')
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])

    plt.title('Degree distribution (log-log scale)')
    plt.xlabel('Node degree')
    plt.ylabel('Probability')
    plt.legend()


def plot_method3(vertex_degree_distribution: (list, list)) -> None:
    """
    Create and display log-log plot & logarithmic bins. This is third part from 4 methods what I want to
    compare in my thesis:
    1. normal - non log scales
    2. log-log - log scales
    3. log-bin - log scales & logarithmic bins
    4. log-log with cumulative adding
    """
    degree, degree_distribution = vertex_degree_distribution
    degree, degree_distribution = log_binning(degree, degree_distribution)
    slope, sigma, degree, degree_distribution_ls = ls_fit(degree, degree_distribution)

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    plt.plot(degree, degree_distribution_ls, 'b', label=f'best fit with slope = {slope:.3f} +- {sigma:.3f}')
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])

    plt.title('Degree distribution (log-log scale)')
    plt.xlabel('Node degree')
    plt.ylabel('Probability')
    plt.legend()


def plot_method4(vertex_degree_distribution: (list, list)) -> None:
    """
    Create and display log-log plot , but with cumulative distribution. This is quasi-best method - CDF also have power
    law, with slope == alpha-1 . This is third part from 4 methods what I want to compare in my thesis:
    1. normal - non log scales
    2. log-log - log scales
    3. log-bin - log scales & logarithmic bins
    4. log-log with cumulative adding
    """
    degree, degree_distribution = vertex_degree_distribution
    degree, degree_distribution = cumulative_binning(degree, degree_distribution)
    slope, sigma, degree, degree_distribution_ls = ls_fit(degree, degree_distribution)

    plt.figure()
    plt.plot(degree, degree_distribution, 'r.', label='simulation')
    plt.plot(degree, degree_distribution_ls, 'b', label=f'best fit with slope = {slope:.3f} +- {sigma:.3f}')
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(degree_distribution[-1], degree_distribution[0])

    plt.title('Cumulative degree distribution (log-log scale)')
    plt.xlabel('Node degree')
    plt.ylabel('Probability')
    plt.legend()
