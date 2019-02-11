import pytest
import numpy as np

from experiment_framework.asymptotic_analysis import compute_bootstrapped_upper_bound

import numpy as np
import pytest

from experiment_framework.asymptotic_analysis import \
    compute_bootstrapped_upper_bound


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
    X = X.astype(np.double)
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
    res = compute_bootstrapped_upper_bound(
        X=X,
        d=3,
        fit_model=fit_model,
        y=y,
        confidence_interval_width=0.1,
        confidence_interval_prob=0.75
    )
    assert res['n_bootstraps'] >= 10
    assert res['k_chebyshev'] >= 3
    assert res['ub'] >= 0.6