import dill
import luigi
import numpy as np
import pytest


from experiment_framework.sampling_framework import GenSamples
from empirical_privacy.laplace_mechanism import (GenSampleLaplaceMechanism,
    EvaluateKNNLaplaceStatDist)
from experiment_framework.helpers import deltas_for_multiple_docs
from experiment_framework.differential_privacy import _ComputeLowerBoundForDelta

@pytest.fixture()
def ds():
    return {
        'database_0' : (0, 0, 0),
        'database_1' : (1, 0, 0),
        'sensitivity': 1.,
        'epsilon'    : 1.,
        'delta'      : 0.
    }


@pytest.fixture()
def asymptotics_settings():
    return {
    'gen_sample_kwargs'  : {'generate_in_batch': True,
                            'x_concatenator': 'numpy.vstack'
                           },
    'fitter'             : 'knn',
    # we use random tie-breaking since the samples are discrete
    'fitter_kwargs'      : {'neighbor_method': 'gyorfi'},
    'n_docs'                : 1,
    'n_trials_per_training_set_size': 5,
    'n_max'              : 2**11,
    'validation_set_size': 2**11,
    'p'                  : 0.9,  # for bootstrap
    't'                  : 0.01  # for bootstrap
}


@pytest.fixture()
def gs(ds):
    gs = GenSampleLaplaceMechanism(
        dataset_settings=ds,
        generate_positive_sample=True,
        random_seed='0'
        )
    return gs


@pytest.fixture()
def clbd(ds, asymptotics_settings) -> _ComputeLowerBoundForDelta:
    CLBDs = deltas_for_multiple_docs(dataset_settings=ds,
                                     GS=GenSampleLaplaceMechanism,
                                     claimed_epsilon=0.1,
                                     **asymptotics_settings)
    CLBD = next(CLBDs)
    return CLBD


def test_gen_sample_laplace_constructor(gs):
    X, y = gs.gen_sample(0)
    assert X.shape == (1, 3)
    assert y == 1


def test_gen_sample_deterministic(gs):
    X, y = gs.gen_sample(0)
    for _ in range(1000):
        Xn, yn = gs.gen_sample(0)
        assert np.allclose(X, Xn) and y == yn


@pytest.mark.parametrize('gen_positive_samples', [True, False])
def test_distinct_samples_generated(gen_positive_samples, gs):
    gen_samples_type = GenSamples(
        gen_sample_type=type(gs),
        x_concatenator='numpy.vstack',
        generate_in_batch=True
        )
    gen_samples_object = gen_samples_type(
        num_samples=64,
        generate_positive_samples=gen_positive_samples,
        dataset_settings={
            'database_0' : (0, 0, 0),
            'database_1' : (1, 0, 0),
            'sensitivity': 1.,
            'epsilon'    : 0.1,
            'delta'      : 0
            },
        random_seed='0'
        )
    luigi.build([gen_samples_object], local_scheduler=True, log_level='ERROR')
    with gen_samples_object.output().open() as f:
        output = dill.load(f)

    # assert all rows are different row-wise
    X = output['X']
    X = X[(X ** 2).sum(axis=1) < 1000, :]
    assert np.linalg.matrix_rank(X) >= 3
    assert np.all(np.abs(np.diff(X, axis=0)) > 0)

@pytest.mark.parametrize('random_seed', np.arange(10))
def test_knn_accuracy_is_at_least_prob_of_alternative_sample(random_seed, gs,
                                                             ds):

    ESD = EvaluateKNNLaplaceStatDist(training_set_size=2**12,
                                     validation_set_size=2**12,
                                     dataset_settings=ds,
                                     random_seed=random_seed)
    luigi.build([ESD], local_scheduler=True, workers=1, log_level='ERROR')
    with ESD.output().open() as f:
        accuracy = dill.load(f)['accuracy']

    # The probability of an alternative sample is
    # 0.5 * gs.probability_of_alternitave_sample, and alternative samples
    # should always be classified correctly
    # the remaining samples should be classified correctly with probability
    # 50% since the underlying distributions are identical
    # allow an envelope of 3% for randomness
    expected_accuracy = 0.5 + 0.5 * gs.probability_of_alternative_sample

    assert expected_accuracy - 0.03 <= accuracy <= expected_accuracy + 0.03


def delta_for_claimed_epsilon(clbd, claimed_epsilon):
    params = clbd.param_kwargs
    params['claimed_epsilon'] = claimed_epsilon
    new_clbd = type(clbd)(**params)
    luigi.build([new_clbd], log_level='ERROR', local_scheduler=True)
    with new_clbd.output().open() as f:
        delta = dill.load(f)
    return delta


def test_delta_with_claimed_epsilon_too_low(ds, clbd):
    delta = delta_for_claimed_epsilon(clbd, ds['epsilon']/2)
    assert delta['lower_bound'] > 0


def test_delta_with_claimed_epsilon_too_high(ds, clbd):
    delta = delta_for_claimed_epsilon(clbd, ds['epsilon']*2)
    assert delta['upper_bound'] < 0


def test_delta_with_claimed_epsilon_just_right(ds, clbd):
    delta = delta_for_claimed_epsilon(clbd, ds['epsilon'])
    assert delta['lower_bound'] < 0 < delta['upper_bound']
