from abc import ABC

import dill
import luigi
from math import exp

from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR, MIN_SAMPLES
from experiment_framework.asymptotic_analysis import _ComputeAsymptoticAccuracy
from experiment_framework.utils.calculations import accuracy_to_statistical_distance
from experiment_framework.utils.luigi_target_mixins import (
    AutoLocalOutputMixin,
    LoadInputDictMixin, DeleteDepsRecursively,
)


class _ComputeBoundsForDelta(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC
):
    claimed_epsilon = luigi.FloatParameter()

    confidence_interval_width = luigi.FloatParameter(default=1.0)
    n_max = luigi.IntParameter()
    min_samples = luigi.IntParameter(default=MIN_SAMPLES)
    dataset_settings = luigi.DictParameter()
    validation_set_size = luigi.IntParameter(default=2 ** 10)
    n_bootstraps = luigi.IntParameter(default=100)
    confidence_interval_prob = luigi.FloatParameter(default=0.90)
    knn_curve_model = luigi.Parameter(default="gyorfi")

    in_memory = luigi.BoolParameter(default=False,
                                    description="Compute dependencies in memory"
                                                " rather than asking the luigi"
                                                " scheduler to compute them.")

    def requires(self):
        reqs = {}
        dataset_settings = dict(self.dataset_settings)
        dataset_settings['claimed_epsilon'] = self.claimed_epsilon
        reqs['asymptotic_accuracy'] = self.asymptotic_accuracy_computer(
            confidence_interval_width=self.confidence_interval_width,
            n_max=self.n_max,
            min_samples=self.min_samples,
            dataset_settings=dataset_settings,
            validation_set_size=self.validation_set_size,
            n_bootstraps=self.n_bootstraps,
            confidence_interval_prob=self.confidence_interval_prob,
            knn_curve_model=self.knn_curve_model,
            in_memory=self.in_memory
        )
        return reqs

    def run(self):
        _inputs = self.load_input_dict()
        statistical_distance = {
            'lower_bound' : accuracy_to_statistical_distance(
                _inputs['asymptotic_accuracy']['lower_bound']),
            'upper_bound' : accuracy_to_statistical_distance(
                _inputs['asymptotic_accuracy']['upper_bound']),
            'lower_bound_slack': accuracy_to_statistical_distance(
                _inputs['asymptotic_accuracy']['lower_bound_slack']),
            'upper_bound_slack': accuracy_to_statistical_distance(
                _inputs['asymptotic_accuracy']['upper_bound_slack']),
            'lb_one_sided': accuracy_to_statistical_distance(
                _inputs['asymptotic_accuracy']['lb_one_sided']),
            'ub_one_sided': accuracy_to_statistical_distance(
                _inputs['asymptotic_accuracy']['ub_one_sided']),
        }
        epsilon = self.claimed_epsilon
        delta = {
            bound_name: compute_delta(stat_dist=sd, epsilon=epsilon) for
            bound_name, sd in statistical_distance.items()
        }
        with self.output().open('wb') as f:
            dill.dump(delta, f)


def compute_delta(stat_dist: float, epsilon: float):
    """implements equation 3 from Yun's notes (Section ?? in the paper)"""
    return exp(epsilon) * (stat_dist - (1 - 1 / exp(epsilon)))


def ComputeBoundsForDelta(asymptotic_accuracy_computer:
_ComputeAsymptoticAccuracy) -> _ComputeBoundsForDelta:
    class T(_ComputeBoundsForDelta):
        pass

    T.asymptotic_accuracy_computer = asymptotic_accuracy_computer

    return T
