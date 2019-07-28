from abc import ABC
from abc import ABC
from collections import namedtuple, defaultdict

import dill
import luigi
import numpy as np

from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR, MIN_SAMPLES, SAMPLES_BASE
from experiment_framework.utils.luigi_target_mixins import AutoLocalOutputMixin, \
    LoadInputDictMixin, \
    DeleteDepsRecursively


def ComputeConvergenceCurve(
        compute_stat_dist: 'EvaluateStatisticalDistance') \
        -> '_ComputeConvergenceCurve':
    class T(_ComputeConvergenceCurve):
        pass

    T.compute_stat_dist = compute_stat_dist
    return T


_CP = namedtuple('CurvePoint', ['trial', 'training_set_size'])


class _ComputeConvergenceCurve(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC
):
    n_max = luigi.IntParameter()
    curve_number = luigi.Parameter()
    min_samples = luigi.IntParameter(default=MIN_SAMPLES)
    dataset_settings = luigi.DictParameter()
    validation_set_size = luigi.IntParameter(default=200)
    in_memory = luigi.BoolParameter(default=False)


    def trial_job(self, trial, training_set_size):
        return self.compute_stat_dist(
            dataset_settings=self.dataset_settings,
            training_set_size=training_set_size,
            validation_set_size=self.validation_set_size,
            random_seed='curve{}_trial{}'.format(self.curve_number, trial),
            in_memory=self.in_memory
        )

    @property
    def priority(self):
        """
        The priority of a particular CCC. Higher is better.
        1. Lower n_max have higher priority
        2. Lower n_trials have higher priority
        """
        return 100 * 1 / self.n_max

    @property
    def pow_min(self):
        return np.floor(np.log(self.min_samples) / np.log(SAMPLES_BASE)
                        + np.spacing(1)).astype(np.int)

    @property
    def n_steps(self):
        pow_max = np.floor(np.log(self.n_max) / np.log(SAMPLES_BASE)
                           + np.spacing(1)).astype(np.int)
        n_steps = pow_max - self.pow_min + 1
        assert n_steps > 0, 'n_steps is {} which results in no ' \
                            'samples.'.format(n_steps)
        return n_steps

    @property
    def _training_set_sizes(self):
        return np.logspace(start=self.pow_min,
                           stop=self.pow_min + self.n_steps - 1,
                           num=self.n_steps,
                           dtype=int, base=SAMPLES_BASE)

    def requires(self):
        reqs = defaultdict(list)
        tss = self._training_set_sizes
        tss_max = tss[-1]
        for ts in tss:
            reqs[ts] = [
                self.trial_job(trial, ts)
                for trial in range(int(tss_max/ts))
            ]
        self.reqs_ = reqs
        if self.in_memory:
            return {}
        return reqs

    def run(self):
        _inputs = self.compute_or_load_requirements()

        self.output_ = {
            'training_set_size_to_accuracy': {ts: [cp['accuracy'] for cp in tscps]
                                              for ts, tscps in _inputs.items()}
        }
        with self.output().open('wb') as f:
            dill.dump(self.output_, f, 2)


def converged(accuracy_list, min_samples=3, convergence_ratio=1e-4):
    if len(accuracy_list) < min_samples:
        return False

    std0 = np.std(accuracy_list[0:2])
    std1 = np.std(accuracy_list[0:3])
    stdn1 = np.std(accuracy_list[0:-1])
    stdn = np.std(accuracy_list)
    d0 = abs(std0 - std1)
    dn = abs(stdn - stdn1)
    #     print(f'd0={d0} dn={dn}')
    return dn / d0 < convergence_ratio
