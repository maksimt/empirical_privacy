import pytest
import numpy as np

from experiment_framework.asymptotic_analysis import empirical_bootstrap_bounds

import numpy as np
import pytest

from experiment_framework.asymptotic_analysis import \
    empirical_bootstrap_bounds


def y_shape():
    return 5, 10

@pytest.fixture()
def y():
    np.random.seed(0)
    y = np.random.randn(*y_shape())*0.1 + 0.5
    y += np.arange(10)/100
    return y

@pytest.fixture()
def X():
    ntri, ntss = y_shape()
    tss = np.logspace(3, 3+ntss, num=ntss, base=2)
    X = np.tile(tss,
                (ntri, 1))
    X = X.astype(np.int)
    return X

@pytest.mark.parametrize('d',[
     3,
     6
 ])
@pytest.mark.parametrize('fit_model', [
    'gyorfi',
    'sqrt_random_tiebreak',
    'sqrt'
])
def test_boot_strap(X, y, fit_model, d):
    res = empirical_bootstrap_bounds(
        training_set_sizes=X,
        d=d,
        fit_model=fit_model,
        classifier_accuracies=y,
        n_bootstraps=10,
        confidence_interval_prob=0.75
    )
    assert res['n_bootstraps'] >= 10
    assert res['lb_two_sided'] <= res['ub_two_sided']
    assert res['lb_two_sided'] <= res['lb_one_sided']
    assert res['ub_one_sided'] <= res['ub_two_sided']
