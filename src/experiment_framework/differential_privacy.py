from abc import ABC
from math import exp

import dill
import luigi

from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR
from experiment_framework.luigi_target_mixins import (AutoLocalOutputMixin,
    LoadInputDictMixin, DeleteDepsRecursively)
from experiment_framework.asymptotic_analysis import _ComputeAsymptoticAccuracy
from experiment_framework.calculations import accuracy_to_statistical_distance


class _ComputeLowerBoundForDelta(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC
):

    claimed_epsilon = luigi.FloatParameter()

    n_trials_per_training_set_size = luigi.IntParameter()
    n_max = luigi.IntParameter()
    dataset_settings = luigi.DictParameter()
    validation_set_size = luigi.IntParameter(default=2 ** 10)
    confidence_interval_width = luigi.FloatParameter(default=0.01)
    confidence_interval_prob = luigi.FloatParameter(default=0.90)

    def requires(self):
        reqs = {}
        reqs['asymptotic_accuracy'] = self.asymptotic_accuracy_computer(
            n_trials_per_training_set_size = self.n_trials_per_training_set_size,
            n_max = self.n_max,
            dataset_settings = self.dataset_settings,
            validation_set_size = self.validation_set_size,
            confidence_interval_width = self.confidence_interval_width,
            confidence_interval_prob = self.confidence_interval_prob
        )
        return reqs

    def run(self):
        _inputs = self.load_input_dict()
        statistical_distance = {
            'lower_bound': accuracy_to_statistical_distance(_inputs['asymptotic_accuracy'][
                                                                'lower_bound']),
            'upper_bound': accuracy_to_statistical_distance(_inputs['asymptotic_accuracy'][
                                                                'upper_bound']),
        }
        epsilon = self.claimed_epsilon
        delta = {
            bound_name: compute_delta(stat_dist=sd, epsilon=epsilon) for
                bound_name, sd in statistical_distance.items()
        }
        with self.output().open('wb') as f:
            dill.dump(delta, f)


def ComputeLowerBoundForDelta(asymptotic_accuracy_computer:
_ComputeAsymptoticAccuracy) -> _ComputeLowerBoundForDelta:
    class T(_ComputeLowerBoundForDelta):
        pass
    T.asymptotic_accuracy_computer = asymptotic_accuracy_computer

    return T

def compute_delta(stat_dist: float, epsilon: float):
    """implements equation 3 from Yun's notes (Section ?? in the paper)"""
    return exp(epsilon) * ( stat_dist - (1 - 1/exp(epsilon)) )
