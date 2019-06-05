import typing
from abc import ABC
from math import sqrt

import dill
import luigi
import numpy as np
from sklearn.utils import resample

from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR
from experiment_framework.luigi_target_mixins import AutoLocalOutputMixin, \
    LoadInputDictMixin, DeleteDepsRecursively
from experiment_framework.privacy_estimator_mixins import get_k
from experiment_framework.sampling_framework import ComputeConvergenceCurve


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
    validation_set_size = luigi.IntParameter(default=2 ** 10)

    n_bootstraps = luigi.IntParameter(default=100)
    confidence_interval_prob = luigi.FloatParameter(default=0.90)

    def requires(self):
        reqs = {}
        reqs['CCC'] = self.CCC(
            n_trials_per_training_set_size=self.n_trials_per_training_set_size,
            n_max=self.n_max,
            dataset_settings=self.dataset_settings,
            validation_set_size=self.validation_set_size
        )
        # a sample is needed for inferring dimension
        reqs['Sample'] = self.CCC.compute_stat_dist.samplegen. \
            gen_sample_type(
            dataset_settings=self.dataset_settings,
            random_seed=0,
            generate_positive_sample=True,
            sample_number=0
        )

        return reqs

    def run(self):
        _inputs = self.load_input_dict()
        res = _inputs['CCC']
        CCC = self.requires()['CCC']
        R1 = list(CCC.requires().values())[0]
        fit_model = R1.requires()['model'].neighbor_method

        y = res['accuracy_matrix']
        # since we sample rows of x, this is equivalent to block bootstrap
        X = np.tile(res['training_set_sizes'],
                    (res['accuracy_matrix'].shape[0], 1))
        X = X.astype(np.double)

        samp = _inputs['Sample'].x
        if samp.ndim == 1:
            d = 1
        else:
            d = samp.shape[1]

        if fit_model == 'gyorfi':
            assert d >= 3, 'The gyorfi convergence rate is only guaranteed to ' \
                           'hold when d>=3, but d={}.'.format(d)

        rtv = empirical_bootstrap_bounds(
            training_set_sizes=X,
            classifier_accuracies=y,
            d=d,
            fit_model=fit_model,
            confidence_interval_prob=self.confidence_interval_prob,
            n_bootstraps=self.n_bootstraps
        )

        rtv['upper_bound'] = rtv['ub_two_sided']
        rtv['lower_bound'] = rtv['lb_two_sided']
        with self.output().open('wb') as f:
            dill.dump(rtv, f)


def ComputeAsymptoticAccuracy(
        compute_convergence_curve: ComputeConvergenceCurve
) -> _ComputeAsymptoticAccuracy:
    class T(_ComputeAsymptoticAccuracy):
        pass

    T.CCC = compute_convergence_curve
    return T


def empirical_bootstrap_bounds(training_set_sizes,
                               d,
                               fit_model,
                               classifier_accuracies,
                               confidence_interval_prob,
                               n_bootstraps):

    k_nearest_neighbors = transform_n_to_k_for_knn(Ns=training_set_sizes,
                                                   fit_model=fit_model,
                                                   d=d)
    classifier_accuracies[classifier_accuracies < 0.5] = 0.5
    # If the classifier is worse than random, the adversary would just guess randomly.
    # This accuracy is a lower bound for the random guess, when priors are
    # 0.5 each. If one class had a larger prior we'd always guess that class.
    sample_mean = asymptotic_privacy_lr(k_nearest_neighbors,
                                        classifier_accuracies)

    delta_means = bootstrap_deltas(classifier_accuracies,
                                   k_nearest_neighbors,
                                   n_bootstraps,
                                   sample_mean,
                                   sample_size=training_set_sizes.shape[0])

    one_sided_error_percentage = (1-confidence_interval_prob) * 100
    two_sided_error_percentage = (1-confidence_interval_prob)/2 * 100

    # eg if confidence_inrvela_prob = 0.9
    # error percentage is 0.1
    # P(mu >= x - d_0.9) >= 0.9
    lb_one_sided = sample_mean - np.percentile(delta_means,
                                               q=100-one_sided_error_percentage,
                                               interpolation='higher')
    # P(mu <= x - d_0.1) >= 0.9
    # error percentage is 0.1
    ub_one_sided = sample_mean - np.percentile(delta_means,
                                               q=one_sided_error_percentage,
                                               interpolation='lower')

    lb_two_sided = sample_mean - np.percentile(delta_means,
                                               q=100 - two_sided_error_percentage,
                                               interpolation='higher')
    ub_two_sided = sample_mean - np.percentile(delta_means,
                                               q=two_sided_error_percentage,
                                               interpolation='lower')

    return {
        'delta_means' : delta_means,
        'n_bootstraps': n_bootstraps,
        'mean'        : sample_mean,
        'lb_one_sided': lb_one_sided,
        'ub_one_sided': ub_one_sided,
        'lb_two_sided': lb_two_sided,
        'ub_two_sided': ub_two_sided
    }


def bootstrap_deltas(classifier_accuracies, k_nearest_neighbors, n_bootstraps, sample_mean,
                     sample_size):
    delta_means = np.empty((n_bootstraps,))
    for i in range(n_bootstraps):
        bootstrap_samples = bootstrap_ci(
            n_samples=sample_size,
            X=k_nearest_neighbors,
            y=classifier_accuracies,
            f=asymptotic_privacy_lr,
            random_state_offset=sample_size * i
        )
        bootstrap_mean = np.mean(bootstrap_samples)
        delta_means[i] = bootstrap_mean - sample_mean
    return delta_means


def bootstrap_ci(n_samples: int, X: np.ndarray, y: np.ndarray,
                 f: typing.Callable[[np.ndarray, np.ndarray, int], np.double],
                 random_state_offset=0) \
        -> np.ndarray:
    """
    Perform block bootstrap row-wise and then reshape into 1-dim arrays

    Examples
    --------
    bootstrap_asymp = bootstrap_ci(100, x, y, asymptotic_curve_torch)

    Parameters
    ----------
    n_samples :
    X :
    y :
    f :
    random_state_offset : int added to random state before setting the seed.

    Returns
    -------

    """
    res = np.empty((n_samples,))
    for tri in range(n_samples):
        Xi, yi = resample(X, y, random_state=tri + random_state_offset)
        res[tri] = f(Xi, yi)
    return res


def asymptotic_privacy_lr(k_nearest_neighbors,
                          classifier_accuracies):
    k_nearest_neighbors = k_nearest_neighbors.reshape(k_nearest_neighbors.size)
    classifier_accuracies = classifier_accuracies.reshape(classifier_accuracies.size)
    b = asymptotic_curve(k_nearest_neighbors, classifier_accuracies)
    return b[0]  # b=[m, C]


def asymptotic_curve(X, y):
    """
    y ~ b[0] + b[1]*X

    Parameters
    ----------
    X :
    y :

    Returns
    -------
    """
    n = X.size
    A = np.ones((n, 2))
    A[:, 1] = X
    fit = np.linalg.lstsq(A, y)
    return fit[0]


def transform_n_to_k_for_knn(Ns, fit_model, d=None):
    if Ns.ndim > 1:
        rtv = np.empty_like(Ns)
        for i in range(Ns.shape[0]):
            rtv[i, :]  = transform_n_to_k_for_knn(Ns[i], fit_model, d)
        return rtv

    if fit_model == 'gyorfi':
        rtv = [-1.0 / get_k(method='gyorfi', num_samples=x, d=d) for x in Ns]
    elif 'sqrt' in fit_model:
        rtv = [-1.0 / get_k(method='sqrt', num_samples=x) for x in Ns]
    return np.array(rtv)


try:
    import torch


    def asymptotic_curve_torch(X: np.ndarray, y: np.ndarray, d: int) -> np.double:
        mod = KNNConvergenceCurve(torch.from_numpy(X), torch.from_numpy(y), d)
        mod.fit_with_optimizer(n_iter=1500)
        return (mod.m.item(), mod.c.item())


    class KNNConvergenceCurve(torch.nn.Module):
        def __init__(self, x: torch.Tensor,
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
