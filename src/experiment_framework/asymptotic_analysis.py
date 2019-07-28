from abc import ABC

import dill
import luigi
import numpy as np

from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR, MIN_SAMPLES
from experiment_framework.compute_convergence_curve import ComputeConvergenceCurve
from experiment_framework.empirical_bootstrap import (
    EmpiricalBootstrap,
    PerTrainingSizeSampleGenerator,
)
from experiment_framework.privacy_estimator_mixins import get_k
from experiment_framework.utils.luigi_target_mixins import (
    AutoLocalOutputMixin,
    LoadInputDictMixin,
    DeleteDepsRecursively,
)


class _ComputeAsymptoticAccuracy(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC,
):
    n_max = luigi.IntParameter()
    min_samples = luigi.IntParameter(default=MIN_SAMPLES)
    dataset_settings = luigi.DictParameter()
    validation_set_size = luigi.IntParameter(default=2 ** 10)

    n_bootstraps = luigi.IntParameter(default=100)
    confidence_interval_prob = luigi.FloatParameter(default=0.90)
    confidence_interval_width = luigi.FloatParameter(default=1.0)

    knn_curve_model = luigi.Parameter(default="gyorfi",
                                      description="The knn curve model."
                                                  "Used to estimate the asymptotic accuracy."
                                                  "'gyorfi' by default.")

    in_memory = luigi.BoolParameter(default=False)

    @property
    def priority(self):
        if 'doc_ind' in self.dataset_settings:
            return 1.0 / (self.dataset_settings['doc_ind'] + 1e-10)
        return 1.0

    def requires(self):
        reqs = {}
        reqs["CCC"] = self.CCC_job(curve_number=0)
        # a sample is needed for inferring dimension
        reqs["Sample"] = self.CCC.compute_stat_dist.samplegen.gen_sample_type(
            dataset_settings=self.dataset_settings,
            random_seed=0,
            generate_positive_sample=True,
            sample_number=0,
        )

        return reqs

    def CCC_job(self, curve_number):
        return self.CCC(
            curve_number=str(curve_number),
            n_max=self.n_max,
            min_samples=self.min_samples,
            dataset_settings=self.dataset_settings,
            validation_set_size=self.validation_set_size,
            in_memory=self.in_memory
        )

    def run(self):
        interval_width = float('inf')
        _inputs = self.load_input_dict()

        fit_model = self.knn_curve_model
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

        n_curves_done = 1
        training_size_to_accuracy = None

        while (n_curves_done < self.min_curves or interval_width > self.confidence_interval_width)\
                and n_curves_done < self.max_curves:

            for _ in range(self.n_curves_per_bootstrap):
                CCC_job = self.CCC_job(curve_number=n_curves_done + 1)
                if not self.in_memory:
                    yield CCC_job
                res = self._populate_obj(CCC_job)
                ts2a = res['training_set_size_to_accuracy']
                if training_size_to_accuracy is None:
                    training_size_to_accuracy = ts2a
                for ts in training_size_to_accuracy.keys():
                    training_size_to_accuracy[ts].extend(ts2a[ts])
                n_curves_done += 1

            if self._skip_bootstrap_because_loading_existing_results(n_curves_done+1):
                continue

            bootstrap = construct_bootstrap(training_size_to_accuracy, d, fit_model)
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

            interval_width = boot_res.ub_two_sided - boot_res.lb_two_sided
            self.set_status_message(
                f"{n_curves_done} curves done | interval_width={interval_width:.3e} "
                f"ratio={interval_width/self.confidence_interval_width:.3f}"
            )

        slack = self.confidence_interval_width - (rtv["upper_bound"] - rtv["lower_bound"])
        rtv["upper_bound_slack"] = rtv["upper_bound"] + slack/2
        rtv["lower_bound_slack"] = rtv["lower_bound"] - slack/2

        rtv['n_curves_done'] = n_curves_done
        with self.output().open("wb") as f:
            dill.dump(rtv, f)

    def _skip_bootstrap_because_loading_existing_results(self, next_job_number):
        next_CCC_job = self.CCC_job(curve_number=next_job_number)
        if hasattr(next_CCC_job, 'complete') and next_CCC_job.complete():
            return True
        return False


def ComputeAsymptoticAccuracy(
        compute_convergence_curve: ComputeConvergenceCurve,
        n_curves_per_bootstrap=10,
        min_curves=10,
        max_curves=3000
) -> _ComputeAsymptoticAccuracy:
    class T(_ComputeAsymptoticAccuracy):
        pass

    T.CCC = compute_convergence_curve
    T.min_curves = min_curves
    T.max_curves = max_curves
    T.n_curves_per_bootstrap = n_curves_per_bootstrap

    return T


def construct_bootstrap(training_size_to_accuracy, d, fit_model):
    knn_to_accuracy = {transform_n_to_k_for_knn(tss, fit_model, d): accu
                for tss, accu in training_size_to_accuracy.items()}
    sample_gen = PerTrainingSizeSampleGenerator(
        data=knn_to_accuracy,
        transform=asymptotic_privacy_lr,
        reshape=get_Xy,
    )
    return EmpiricalBootstrap(sample_generator=sample_gen)


def get_Xy(tss_to_accuracy):
    X, y = list(), list()
    for tss, accuracies in tss_to_accuracy.items():
        X.extend([tss for _ in accuracies])
        y.extend(accuracies)
    X = np.array(X)
    y = np.array(y)
    X = X.astype(np.double)
    return X, y


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


def transform_n_to_k_for_knn(n, fit_model, d=None):
    if fit_model == "gyorfi":
        rtv = -1.0 / n ** (1 / (d + 2))  # 1/(d+2) is from Doring
    elif "sqrt" in fit_model:
        rtv = -1.0 / get_k(method="sqrt", num_samples=n)
    return rtv


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
