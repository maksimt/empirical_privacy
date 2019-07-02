from abc import ABC

import dill
import luigi
import numpy as np

from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR
from experiment_framework.empirical_bootstrap import (
    EmpiricalBootstrap,
    SampleGenerator
)
from experiment_framework.utils.luigi_target_mixins import (
    AutoLocalOutputMixin,
    LoadInputDictMixin,
    DeleteDepsRecursively,
)
from experiment_framework.privacy_estimator_mixins import get_k
from experiment_framework.sampling_framework import ComputeConvergenceCurve


class _ComputeAsymptoticAccuracy(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC,
):
    n_trials_per_training_set_size = luigi.IntParameter()
    n_max = luigi.IntParameter()
    dataset_settings = luigi.DictParameter()
    validation_set_size = luigi.IntParameter(default=2 ** 10)

    n_bootstraps = luigi.IntParameter(default=100)
    confidence_interval_prob = luigi.FloatParameter(default=0.90)
    knn_curve_model = luigi.Parameter(default="gyorfi",
                                      description="The knn curve model."
                                                  "Used to estimate the asymptotic accuracy."
                                                  "'gyorfi' by default.")

    in_memory = luigi.BoolParameter(default=False)

    def requires(self):
        reqs = {}
        reqs["CCC"] = self.CCC(
            n_trials_per_training_set_size=self.n_trials_per_training_set_size,
            n_max=self.n_max,
            dataset_settings=self.dataset_settings,
            validation_set_size=self.validation_set_size,
            in_memory=self.in_memory
        )
        # a sample is needed for inferring dimension
        reqs["Sample"] = self.CCC.compute_stat_dist.samplegen.gen_sample_type(
            dataset_settings=self.dataset_settings,
            random_seed=0,
            generate_positive_sample=True,
            sample_number=0,
        )

        return reqs

    def run(self):
        _inputs = self.load_input_dict()
        res = _inputs["CCC"]
        fit_model = self.knn_curve_model

        y = res["accuracy_matrix"]
        # since we sample rows of x, this is equivalent to block bootstrap
        X = np.tile(res["training_set_sizes"], (res["accuracy_matrix"].shape[0], 1))
        X = X.astype(np.double)

        samp = _inputs["Sample"].x
        if samp.ndim == 1:
            d = 1
        else:
            d = samp.shape[1]

        if fit_model == "gyorfi":
            assert d >= 3, (
                "The gyorfi convergence rate is only guaranteed to "
                "hold when d>=3, but d={}.".format(d)
            )

        bootstrap = construct_bootstrap(X, d, fit_model, y)
        boot_res = bootstrap.bootstrap_confidence_bounds(
            self.confidence_interval_prob,
            n_samples=self.n_bootstraps
        )
        rtv = {}
        rtv["upper_bound"] = boot_res.ub_two_sided
        rtv["lower_bound"] = boot_res.lb_two_sided

        rtv["lb_one_sided"] = boot_res.lb_one_sided
        rtv["ub_one_sided"] = boot_res.ub_one_sided
        rtv["lb_two_sided"] = boot_res.lb_two_sided
        rtv["ub_two_sided"] = boot_res.ub_two_sided

        with self.output().open("wb") as f:
            dill.dump(rtv, f)


def ComputeAsymptoticAccuracy(
    compute_convergence_curve: ComputeConvergenceCurve
) -> _ComputeAsymptoticAccuracy:
    class T(_ComputeAsymptoticAccuracy):
        pass

    T.CCC = compute_convergence_curve
    return T


def construct_bootstrap(X, d, fit_model, classifier_accuracies):
    k_nearest_neighbors = transform_n_to_k_for_knn(
        Ns=X, fit_model=fit_model, d=d
    )
    classifier_accuracies[classifier_accuracies < 0.5] = 0.5
    data = np.array([asymptotic_privacy_lr([ks, accus])
                     for ks, accus in zip(k_nearest_neighbors, classifier_accuracies)])
    sample_gen = SampleGenerator(data=data)
    return EmpiricalBootstrap(sample_generator=sample_gen)


def asymptotic_privacy_lr(input_):
    k_nearest_neighbors, classifier_accuracies = input_
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
            rtv[i, :] = transform_n_to_k_for_knn(Ns[i], fit_model, d)
        return rtv

    if fit_model == "gyorfi":
        rtv = [-1.0 / x**(1/(d+2)) for x in Ns]  # 1/(d+2) is from Doring
    elif "sqrt" in fit_model:
        rtv = [-1.0 / get_k(method="sqrt", num_samples=x) for x in Ns]
    return np.array(rtv)


try:
    import torch

    def asymptotic_curve_torch(X: np.ndarray, y: np.ndarray, d: int) -> np.double:
        mod = KNNConvergenceCurve(torch.from_numpy(X), torch.from_numpy(y), d)
        mod.fit_with_optimizer(n_iter=1500)
        return (mod.m.item(), mod.c.item())

    class KNNConvergenceCurve(torch.nn.Module):
        def __init__(
            self, x: torch.Tensor, y: torch.Tensor, d: int, print_to_console=False
        ):
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

        def fit_with_optimizer(
            self,
            learning_rate=0.01,
            n_iter=500,
            opt=torch.optim.Adam,
            loss_fn=torch.nn.MSELoss(reduction="sum"),
        ):
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
