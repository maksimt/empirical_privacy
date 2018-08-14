from scipy.stats import binom, norm
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from joblib import Memory, Parallel, delayed, cpu_count
import os
from math import sqrt, ceil, log
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import lsdd
from collections import Mapping

memory = Memory(cachedir=os.path.expanduser('.'), verbose=0)
x_axis = None
n_jobs = cpu_count()/2 - 1

def gen_data(n, p, num_samples, trial, type='binom'):

    np.random.seed(trial)
    if type=='binom':
        B0 = binom.rvs(n - 1, p, size=num_samples) + 0
        B1 = binom.rvs(n - 1, p, size=num_samples) + 1
    elif type=='norm':
        sigma = sqrt((n - 0.75) / 12.0)
        mu = (n - 1.0) / 2
        B0 = norm.rvs(loc=mu+0.25, scale=sigma, size=num_samples)
        B1 = norm.rvs(loc=mu+0.75, scale=sigma, size=num_samples)

    X0 = B0[:, np.newaxis]# = np.concatenate((B0, B1))[:, np.newaxis]
    X1 = B1[:, np.newaxis]
    y0 = np.zeros((num_samples,))
    y1 = np.ones((num_samples,))
    return X0, X1, y0, y1

def get_lsdd_correctness_rate(n, p, num_samples, trial, type='binom'):
    X0, X1, y0, y1 = gen_data(n, p, num_samples, trial, type)
    X0 = X0.ravel()
    X1 = X1.ravel()
    rtv = lsdd.lsdd(X0[np.newaxis, :], X1[np.newaxis, :])
    sd_est = np.mean(np.abs(rtv[1]))
    return 0.5 + 0.5*rtv[0]
get_lsdd_correctness_rate_cached = memory.cache(get_lsdd_correctness_rate)


def get_density_est_correctness_rate(n, p, num_samples, trial,
                                     bandwidth_method=None, type='binom'):
    """

    Parameters
    ----------
    n :
    p :
    num_samples :
    trial :
    bandwidth_method : function, float, or 'cv'
        How should bandwidth be chosen
        If a function f, bandwidth = f(n)/n^(1/d)
        If a float e, bandwidth = 1/n^(1/d-e)
        If cv cross validation is performed

    Returns
    -------

    """
    X0, X1, y0, y1 = gen_data(n, p, num_samples, trial, type)
    X0 = X0.ravel()
    X1 = X1.ravel()
    bw = None
    if hasattr(bandwidth_method, '__call__'):
        bw = float(bandwidth_method(num_samples)) / num_samples  # eg log
    if hasattr(bandwidth_method, '__add__'): # float:
        bw = num_samples ** (1 - bandwidth_method)

    f0 = gaussian_kde(X0, bw_method=bw)
    f1 = gaussian_kde(X1, bw_method=bw)
    #Omega = np.unique(np.concatenate((X0, X1)))
    _min = 0
    _max = n
    #x = np.linspace(_min, _max, num=10*num_samples)
    return 0.5 + \
           0.5 * 0.5 * quad(lambda x: np.abs(f0(x)-f1(x)), -np.inf, np.inf)[0]

get_density_est_correctness_rate_cached = memory.cache(
    get_density_est_correctness_rate
)

def get_expectation_correctness_rate(
    n, p, num_samples, trial,  bandwidth_method, type='binom'
):
    """

    Parameters
    ----------
    n :
    p :
    num_samples :
    trial :
    bandwidth_method : function, float, or 'cv'
        How should bandwidth be chosen
        If a function f, bandwidth = f(n)/n^(1/d)
        If a float e, bandwidth = 1/n^(1/d-e)
        If cv cross validation is performed

    Returns
    -------

    """
    X0, X1, y0, y1 = gen_data(n, p, num_samples, trial, type)
    X0 = X0.ravel()
    X1 = X1.ravel()
    bw = None
    if hasattr(bandwidth_method, '__call__'):
        bw = float(bandwidth_method(num_samples)) / num_samples  # eg log
    if hasattr(bandwidth_method, '__add__'): # float
        bw = num_samples ** (1 - bandwidth_method)

    f0 = gaussian_kde(X0, bw_method=bw)
    f1 = gaussian_kde(X1, bw_method=bw)
    #Omega = np.unique(np.concatenate((X0, X1)))
    _min = 0
    _max = n
    X = np.concatenate((X0, X1))
    f0x = f0(X)
    f1x = f1(X)
    denom = (f0x + f1x + np.spacing(1))
    numer = np.abs(f0x - f1x)
    return 0.5 + 0.5 * np.mean(numer / denom)

get_expectation_correctness_rate_cached = memory.cache(
    get_expectation_correctness_rate
)


def get_knn_correctness_rate(n, p, num_samples, trial, neighbor_method,
                             type='binom'):
    X0, X1, y0, y1 = gen_data(n, p, num_samples, trial, type)
    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))
    KNN = neighbors.KNeighborsClassifier(algorithm='brute', metric='l2')
    if hasattr(neighbor_method, 'lower'):  # string
        if neighbor_method == 'sqrt':
            k = int(ceil(sqrt(num_samples)))
            if k%2==0:  # ensure k is odd
                k += 1
            KNN.n_neighbors = k
        if neighbor_method == 'cv':
            param_grid =\
                [{'n_neighbors':(num_samples ** np.linspace(0.1,1,9)).astype(np.int)}]
            gs = GridSearchCV(KNN, param_grid, scoring=make_scorer(accuracy_score),
                              cv=min([3, num_samples]))
            gs.fit(X, y)
            KNN = gs.best_estimator_
        if neighbor_method == 'sqrt_random_tiebreak':
            k = int(ceil(sqrt(num_samples)))
            if k % 2 == 0:  # ensure k is odd
                k += 1
            KNN.n_neighbors = k
            X = X + np.random.rand(X.shape[0], X.shape[1])*0.1


    KNN.fit(X, y)
    return KNN.score(X, y)
get_knn_correctness_rate_cached = memory.cache(get_knn_correctness_rate)

#@memory.cache
def _get_correctness(n, p, max_N, trial, alg_name, alg_kwargs):
    N = int(ceil(log(max_N) / log(2)))
    N_samples = np.logspace(4,N,num=N-3, base=2).astype(np.int)
    #alg_name = eval(alg_name)
    rtv = [alg_name(n=n, p=p, num_samples=ns, trial=trial, **alg_kwargs) \
           for ns in N_samples]
    return np.array(rtv)[np.newaxis, :]

def get_res(n, p, n_tri, alg, alg_kwargs, n_max = 2**10):
    Trials = np.arange(n_tri)

    res = Parallel(n_jobs=1, verbose=11)(
        delayed(_get_correctness)(n, p, n_max, tr,
                                  alg,
                                  alg_kwargs
                                 )
        for tr in Trials)
    res = reduce(lambda x, y: np.vstack((x, y)), res)
    return res