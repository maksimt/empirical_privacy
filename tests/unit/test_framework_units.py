import os
from glob import glob
from itertools import product

import dill
import luigi
import numpy as np
import pytest

from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR
from empirical_privacy.one_bit_sum import (
    GenSamplesOneBit,
    FitKNNModelOneBit,
    EvaluateKNNOneBitSD,
    ComputeOneBitKNNConvergence,
    OneBitAsymptoticAccuracy,
)


def ds():
    n = 40
    p = 0.5
    return {
        'n_trials'      : n,
        'prob_success'  : p,
        'gen_distr_type': 'binom',
    }


def get_res(job):
    cleanup()
    luigi.build([job], local_scheduler=True, workers=1, log_level='ERROR')
    with job.output().open() as f:
        res = dill.load(f)
        return res


def cleanup():
    for fn in glob(os.path.join(LUIGI_COMPLETED_TARGETS_DIR, '*')):
        try:
            os.remove(fn)
        except (IsADirectoryError, FileNotFoundError):
            pass
    os.sync()


@pytest.mark.parametrize('random_seed', [str(i) for i in range(1336, 1339)])
@pytest.mark.parametrize('gen_pos', [True, False])
def test_gen_samples_deterministic(random_seed, gen_pos):
    jobs = (GenSamplesOneBit(dataset_settings=ds(),
                             random_seed=random_seed,
                             generate_positive_samples=gen_pos,
                             num_samples=2 ** 8
                             ) for _ in range(4))
    res = []
    for (i, job) in enumerate(jobs):
        if i % 2 == 0:
            job.generate_in_batch = False
        else:
            job.generate_in_batch = True
        res.append(get_res(job))

    assert len(res) >= 2
    for i in range(1, len(res)):
        assert np.allclose(res[i - 1]['X'], res[i]['X'])


@pytest.mark.parametrize('random_seed', [str(i) for i in range(1336, 1345)])
def test_fit_model_deterministic(random_seed):
    jobs = [FitKNNModelOneBit(dataset_settings=ds(),
                              random_seed=random_seed,
                              samples_per_class=2 ** 7,
                              in_memory=in_memory)
            for (in_memory, _) in product([True, False], range(3))
            ]
    res = []
    for job in jobs:
        res.append(get_res(job))

    assert len(res) >= 2
    for i in range(1, len(res)):
        assert np.allclose(res[i - 1]['KNN']._fit_X, res[i]['KNN']._fit_X)
        assert np.allclose(res[i - 1]['KNN']._y, res[i]['KNN']._y)


@pytest.mark.parametrize('random_seed', [str(i) for i in range(1336, 1339)])
@pytest.mark.parametrize('training_set_size', [2 ** 7, 2 ** 8])
@pytest.mark.parametrize('validation_set_size', [2 ** 7, 2 ** 8])
def test_eval_stat_dist_deterministic(random_seed,
                                      training_set_size,
                                      validation_set_size):
    jobs = [
        EvaluateKNNOneBitSD(
            dataset_settings=ds(),
            random_seed=random_seed,
            training_set_size=training_set_size,
            validation_set_size=validation_set_size,
            in_memory=in_memory
        ) for (in_memory, _) in product([True, False], range(3))
    ]
    res = []
    for job in jobs:
        res.append(get_res(job))

    assert len(res) >= 2
    for i in range(1, len(res)):
        assert res[i - 1] == res[i]


@pytest.mark.parametrize('random_seed', [str(i) for i in range(1336, 1339)])
def test_ccc_deterministic(random_seed, ):
    jobs = [
        ComputeOneBitKNNConvergence(
            dataset_settings=ds(),
            n_max=2 ** 8,
            n_trials_per_training_set_size=5,
            validation_set_size=2 ** 8,
            in_memory=in_memory
        ) for (in_memory, _) in zip([True, False], range(3))
    ]
    res = []
    for job in jobs:
        res.append(get_res(job))
    assert len(res) >= 2
    for i in range(1, len(res)):
        np.allclose(res[i - 1]['accuracy_matrix'], res[i]['accuracy_matrix'])


def test_bootstrap_deterministic():
    jobs = [
        OneBitAsymptoticAccuracy(
            dataset_settings=ds(),
            n_max=2 ** 8,
            n_trials_per_training_set_size=5,
            validation_set_size=2 ** 8,
            n_bootstraps=30,
            confidence_interval_prob=0.9,
            knn_curve_model='sqrt',
            in_memory=in_memory
        ) for (in_memory, _) in product([True, False], range(3))
    ]
    res = []
    for job in jobs:
        res.append(get_res(job))
    assert len(res) >= 2
    for i in range(1, len(res)):
        res[i - 1] == res[i]
