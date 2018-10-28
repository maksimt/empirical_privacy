import pytest

from empirical_privacy import one_bit_sum_joblib


@pytest.fixture(scope='session')
def ds_rs():
    return {
        'dataset_settings': {
            'n_trials'      : 40, 'prob_success': 0.5,
            'gen_distr_type': 'binom'
        },
        'random_seed'     : '1338'
    }


def test_pytest():
    assert 3 == (2 + 1)


def test_import():
    X0, X1, y0, y1 = one_bit_sum_joblib.gen_data(10, 0.5, 100, 0)
    assert X0.shape[0] == 100