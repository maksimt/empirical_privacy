from math import ceil
import typing
from abc import ABC
from functools import partial

import dill
import luigi
import numpy as np
from sklearn.utils import resample
from scipy.optimize import least_squares

from experiment_framework.sampling_framework import ComputeConvergenceCurve
from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR
from experiment_framework.luigi_target_mixins import AutoLocalOutputMixin, \
    LoadInputDictMixin, DeleteDepsRecursively
from experiment_framework.privacy_estimator_mixins import get_k


class _ComputeAsymptoticAccuracy(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC
    ):

    n_trials_per_training_set_size = luigi.IntParameter()
    n_max = luigi.IntParameter()
    dataset_settings = luigi.DictParameter()
    validation_set_size = luigi.IntParameter(default=2**10)

    confidence_interval_width = luigi.FloatParameter(default=0.01)
    confidence_interval_prob = luigi.FloatParameter(default=0.99)

    def requires(self):
        reqs = {}
        reqs['CCC'] = self.CCC(
            n_trials_per_training_set_size=self.n_trials_per_training_set_size,
            n_max = self.n_max,
            dataset_settings=self.dataset_settings,
            validation_set_size=self.validation_set_size
       )
        # a sample is needed for inferring dimension
        reqs['Sample'] = self.CCC.compute_stat_dist.samplegen. \
            gen_sample_type(
            dataset_settings = self.dataset_settings,
            random_seed = 0,
            generate_positive_sample = True,
            sample_number = 0
        )

        return reqs

    def run(self):
        _inputs = self.load_input_dict()
        res = _inputs['CCC']
        y = res['sd_matrix']
        # since we sample rows of x, this is equivalent to block bootstrap
        x = np.tile(res['training_set_sizes'],
                    (res['sd_matrix'].shape[0], 1))
        x = x.astype(np.double)

        samp = _inputs['Sample'].x
        if samp.ndim==1:
            d = 1
        else:
            d = samp.shape[1]

        # assert d>=3, 'This convergence rate is only guaranteed to hold when ' \
        #              'd>=3, but d={}'.format(d)

        n_bootstraps = hoeffding_n_given_t_and_p(
            t=self.confidence_interval_width,
            p=self.confidence_interval_prob
        )

        bootstrap_samples = bootstrap_ci(
            n_bootstraps,
            X=x,
            y=y,
            f=partial(asymptotic_privacy_lr, d=d)
        )

        rtv = {
            'mean' : np.mean(bootstrap_samples),
            'median' : np.median(bootstrap_samples),
            'std': np.std(bootstrap_samples),
            'n_bootstraps': n_bootstraps,
            'p' : self.confidence_interval_prob
            }
        rtv['upper_bound'] = rtv['mean'] + self.confidence_interval_width
        with self.output().open('wb') as f:
            dill.dump(rtv, f)


def ComputeAsymptoticAccuracy(
        compute_convergence_curve: ComputeConvergenceCurve
        ) -> _ComputeAsymptoticAccuracy:
    class T(_ComputeAsymptoticAccuracy):
        pass

    T.CCC = compute_convergence_curve
    return T


def hoeffding_n_given_t_and_p(t:np.double, p:np.double, C=0.5) -> int:
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


def asymptotic_privacy_lr(X, y, d=6):
    b = asymptotic_curve_sqrt_lr(X, y)
    return b[0]


def asymptotic_curve_sqrt_lr(X, y):
    """
    y ~ b[0] - b[1]*odd(floor(sqrt(X)))

    Parameters
    ----------
    X :
    y :

    Returns
    -------
    """
    n = X.size
    A = np.ones((n, 2))
    A[:, 1] = [get_k(method='sqrt', num_samples=x) for x in X]
    fit = np.linalg.lstsq(A, y)
    return fit[0]


def asymptotic_curve_gyorfi_lr(X, y, d=6):
    """
    y ~ b[0] + X**(-2/(d+2))*b[1]

    Parameters
    ----------
    X :
    y :
    d :

    Returns
    -------

    """
    n = X.size
    A = np.ones((n, 2))
    A[:, 1] = X**(-2/(d+2))
    fit = np.linalg.lstsq(A, y)
    return fit[0]


def constrained_curve(x, y, d):
    lsq = LSQ(x, y, d)
    x0 = asymptotic_curve_gyorfi_lr(x, y, d)[0]
    x0[0] = np.max(x) + 0.01
    lb = np.array([np.max(y), -np.inf])
    ub = np.array([np.inf, np.inf])
    rtv = least_squares(lsq.residuals, x0,
                        bounds=(lb, ub))
    return rtv.x

class LSQ:
    """Hold data so that least-squares residuals can be computed efficiently"""
    def __init__(self, x, y, d):
        n = x.size
        self.A = np.ones((n, 2))
        self.A[:, 1] = x ** (-2 / (d + 2))
        self.y = y

    def residuals(self, b):
        return (np.dot(self.A, b) - self.y) ** 2



def bootstrap_ci(n_samples: int, X: np.ndarray, y: np.ndarray,
                 f: typing.Callable[[np.ndarray, np.ndarray, int], np.double]) \
        -> np.ndarray:
    """

    Examples
    --------
    bootstrap_asymp = bootstrap_ci(100, x, y, asymptotic_curve_torch)

    Parameters
    ----------
    n_samples :
    X :
    y :
    f :

    Returns
    -------

    """
    res = np.empty((n_samples,))
    for tri in range(n_samples):
        Xi, yi = resample(X, y, random_state=tri)
        Xi = Xi.reshape(Xi.size)
        yi = yi.reshape(yi.size)
        res[tri] = f(Xi, yi)
    return res

try:
    import torch

    def asymptotic_curve_torch(X: np.ndarray, y: np.ndarray, d: int) -> np.double:
        mod = KNNConvergenceCurve(torch.from_numpy(X), torch.from_numpy(y), d)
        mod.fit_with_optimizer(n_iter=1500)
        return (mod.m.item(), mod.c.item())

    class KNNConvergenceCurve(torch.nn.Module):
        def __init__(self,  x: torch.Tensor,
                            y: torch.Tensor,
                            d: int,
                            print_to_console=False):
            super(KNNConvergenceCurve, self).__init__()
            self.m = torch.ones(1, requires_grad=True, dtype=torch.double)
            self.c = torch.ones(1, requires_grad=True, dtype=torch.double)
            self.x = x
            self.y = y
            self.x.requires_grad = False
            self.y.requires_grad = False
            self.d = d
            self.print_to_console = print_to_console

        def predict(self, x: int) -> np.double:
            """
            Return asymptotic estimate of error given x training samples

            Parameters
            ----------
            x : int
                The number of data samples

            Returns
            -------
            y : double

            """
            return self.m + self.c * 1 / (x ** (2 / (self.d + 2)))

        def loss(self, x: int) -> np.double:
            return (self.y - self.predict(x)).abs().sum()

        def fit(self, learning_rate=0.01, n_iter=500):
            for t in range(n_iter):
                loss = self.loss(self.x)
                loss.backward()
                with torch.no_grad():
                    self.m -= learning_rate * self.m.grad
                    self.c -= learning_rate * self.c.grad
                    self.m.grad.zero_()
                    self.c.grad.zero_()

        def fit_with_optimizer(self,
                               learning_rate=0.01,
                               n_iter=500,
                               opt=torch.optim.Adam,
                               loss_fn=torch.nn.MSELoss(reduction='sum')):
            opt = opt([self.m, self.c], lr=learning_rate)
            for t in range(n_iter):
                y_pred = self.predict(self.x)
                loss = loss_fn(y_pred, self.y)
                if self.print_to_console and t % 25 == 0:
                    print(t, loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
except ImportError:
    pass