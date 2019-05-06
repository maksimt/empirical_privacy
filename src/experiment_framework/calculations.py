from math import ceil, sqrt

import numpy as np


def hoeffding_n_given_t_and_p_one_sided(t:np.double, p:np.double, C=0.5) -> int:
    """
    Return n such that with probability at least p, P(E[X] < \bar X_n + t).

    Where \bar X_n is the mean of n samples.

    Parameters
    ----------
    t : double
        one sided confidence interval width
    p : double
        probability of bound holding
    C : double
        Width of sample support domain. E.g. 0.5 if all samples fall in
            [0.5, 1.0]

    Returns
    -------

    """
    return int(ceil(C ** 2 * np.log(1 - p) / (-2 * t ** 2)))


def hoeffding_n_given_t_and_p_two_sided(t:np.double, p:np.double, C=0.5) -> int:
    """
    Return n such that with probability at least p, P(|E[X] - \bar X_n| <= t).

    Where \bar X_n is the mean of n samples.

    Parameters
    ----------
    t : double
        two sided confidence interval width
    p : double
        probability of bound holding
    C : double
        Width of sample support domain. E.g. 0.5 if all samples fall in
            [0.5, 1.0]

    Returns
    -------

    """
    return int(ceil(C ** 2 * np.log( 0.5*(1 - p) ) / (-2 * t ** 2)))


def chebyshev_k_from_upper_bound_prob(p_bound_holds:np.double) -> int:
    """
    Return k such with with probability at least p_bound_holds X will be < mu + k*sigma

    Parameters
    ----------
    p_bound_holds : double

    Returns
    -------

    """
    p_bound_violated = 1 - p_bound_holds
    return int(ceil(sqrt(1 / p_bound_violated)))


def accuracy_to_statistical_distance(accuracy):
    return (accuracy - 0.5) * 2


def statistical_distance_to_accuracy(statistical_distance):
    return 0.5 + 0.5 * statistical_distance