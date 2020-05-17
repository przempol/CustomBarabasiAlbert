import numpy as np
from scipy.special import gamma, zeta


def solution_mft(k: int, alpha: float, m: int = 1) -> float:
    """
        Functions that return value of probability that was calculated analytical using mean-field theory and stand for
        if alpha!=0

        (m-2*(1-alpha)/alpha) ^ 2 * 2/alpha * (k + 2 - 2 alpha)^(-2-alpha),

        note that this equation will broke at alpha -> 0, thus if alpha=0 :
        e/m * exp(-k/m)

        :param k: vertex degree
        :param alpha: alpha value
        :param m: at this moment always equal 1, originally this is yet another generalization of BA graph
        :return: probability of given vertex degree
    """
    if alpha > 0.001:
        m = 1
        ret = 2
        ret *= m ** (2 / alpha)
        ret *= (2 - alpha) ** (2 / alpha)
        ret /= alpha ** (1 + 2 / alpha)
        ret *= (k - 2 * m * ((alpha - 1) / alpha)) ** (-(alpha + 2) / alpha)
    else:
        m = 1
        ret = np.e / m * np.exp(- k / m)
    return ret


def solution_me(k: int, alpha: float, m: int = 1) -> float:
    """
        Functions that return value of probability that was calculated analytical using Master equation and stand for
        if alpha!=0

        P(k) = bunch of gamma functions...

        note that this equation will broke at alpha -> 0, thus if alpha == 0 :
        m^(k-m) / (m+1)^(k-m+1)

        :param k: vertex degree
        :param alpha: alpha value
        :param m: at this moment always equal 1, originally this is yet another generalization of BA graph
        :return: probability of given vertex degree
    """
    if alpha > 0.001:
        np.seterr(divide='ignore', invalid='ignore')
        m = 1
        ret = 2 / (2 * m + 2 - alpha * m)
        ret *= gamma(2 * m / alpha - m + 1 + 2 / alpha) / gamma(2 * m / alpha - m)
        # ret /= gamma(2 * m / alpha - m)
        ret *= gamma(2 * m / alpha - 2 * m + k) / gamma(2 * m / alpha - 2 * m + 1 + 2 / alpha + k)
        # ret /= gamma(2 * m / alpha - 2 * m + 1 + 2/alpha + k)
        np.seterr(all=None, divide=None, over=None, under=None, invalid=None)
    else:
        m = 1
        ret = m ** (k - m) / (m + 1) ** (k - m + 1)
    return ret


def ls_fit(x: list, y: list) -> (float, float, list, list):
    """
    Function computing alpha and sigma_alpha using least square method, on logarithmic data.
    NOTE: DATA is not logarithmic! Function is doing it itself.

    :param x: list of degrees
    :param y: list of degrees distribution
    :return: predicted alpha, and x (degree list) alongside with predicted y (degrees distribution list)
    """
    x_true = np.log(x)
    y_true = np.log(y)

    coeff, cov = np.polyfit(x_true, y_true, 1, cov=True)
    f = np.poly1d(coeff)
    # calculate new x's and y's
    y_ret: list = f(x_true)
    y_ret = np.exp(y_ret)
    x_ret = np.exp(x_true)
    return coeff[0], np.sqrt(np.diag(cov))[0], x_ret, y_ret


def mlm_fit(vertex_degrees: list, x_min: int) -> (float, float):
    """
    Function computing alpha and sigma_alpha using maximum likelihood method

    :param vertex_degrees: list of vertex degrees
    :param x_min: the value of x, where we can say that for x >= x_min is power law distribution
    :return: predicted alpha, and x (degree list) alongside with predicted y (degrees distribution list)
    """
    vertex_degrees = np.array(vertex_degrees)
    vertex_degrees = vertex_degrees[vertex_degrees >= x_min]
    vertex_degrees = np.asarray(vertex_degrees) / x_min
    vertex_degrees_log = np.log(vertex_degrees)
    alpha = 1 + len(vertex_degrees_log) / vertex_degrees_log.sum()
    sigma = (alpha - 1) / np.sqrt(len(vertex_degrees_log))
    print(f'Best fit, using x_min = {x_min} is alpha = {alpha:.3f} +- {sigma:.3f}')
    return alpha, sigma


def log_binning(degree: list, degree_distribution: list) -> (list, list):
    bins = np.logspace(0, np.log10(np.max(degree)), int(np.log10(np.max(degree)) * 4 + 1))
    values = np.array([])
    for ii in range(len(bins) - 1):
        mtx = np.where((degree < bins[ii + 1] + 0.001) & (degree >= bins[ii]))
        values = np.append(values, np.sum(degree_distribution[mtx]))

    width = bins[1:] - bins[:-1]
    values = values / width

    bins = (bins[1:] + bins[:-1]) / 2

    zeros = np.where(values == 0.)
    bins = np.delete(bins, zeros)
    values = np.delete(values, zeros)
    values = values / np.sum(degree_distribution)
    return bins, values


def cumulative_binning(degree: list, degree_distribution: list) -> (list, list):
    ret = np.flipud(degree_distribution)
    ret = np.cumsum(ret)
    ret = np.flipud(ret)
    ret = ret / np.sum(degree_distribution)
    return degree, ret


def pmf_power_law(x: int, alpha: float) -> float:
    ret = x ** (-alpha)
    ret /= zeta(alpha, 1)
    return ret


def generate_power_law(x: np.int, alpha: np.float) -> np.float:
    ret = 1 - x
    ret = ret ** (-1 / (alpha - 1))
    # ret = np.int(ret)
    return ret


def random_discrete_power_law(alpha: float) -> int:
    x = 1
    suma = pmf_power_law(x, alpha)
    rnd = np.random.rand()
    while suma < rnd:
        x += 1
        suma += pmf_power_law(x, alpha)
    return x
