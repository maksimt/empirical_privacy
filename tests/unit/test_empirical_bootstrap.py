import numpy as np
import pytest
from scipy.stats import norm

from experiment_framework.empirical_bootstrap import EmpiricalBootstrap,\
    SampleGenerator, TransformingSampleGenerator


def y_shape():
    return 5, 10


@pytest.fixture()
def y():
    np.random.seed(0)
    y = np.random.randn(*y_shape()) * 0.1 + 0.5
    y += np.arange(10) / 100
    return y


@pytest.fixture()
def X():
    ntri, ntss = y_shape()
    tss = np.logspace(3, 3 + ntss, num=ntss, base=2)
    X = np.tile(tss,
                (ntri, 1))
    X = X.astype(np.int)
    return X


@pytest.mark.parametrize('random_seed', range(3))
@pytest.mark.parametrize('bootstrap_size', [100, 1000])
@pytest.mark.parametrize('confidence_interval_probability', [0.9, 0.99])
def test_confidence_interval_bounds(random_seed,
                                    bootstrap_size,
                                    confidence_interval_probability):
    np.random.seed(random_seed)
    samples = np.random.randn(100)
    EB = EmpiricalBootstrap(SampleGenerator(samples))
    rtv = EB.bootstrap_confidence_bounds(confidence_interval_probability,
                                         bootstrap_size)
    # the two sided bound should be looser because there's less probability
    # assigned to the tails
    assert rtv.lb_two_sided <= rtv.ub_two_sided
    assert rtv.lb_two_sided <= rtv.lb_one_sided
    assert rtv.ub_one_sided <= rtv.ub_two_sided


@pytest.mark.parametrize('random_seed', range(3))
@pytest.mark.parametrize('bootstrap_size', [100, 1000])
@pytest.mark.parametrize('confidence_interval_probability', [0.9, 0.99])
def test_bootstrap_implementation(random_seed,
                                  bootstrap_size,
                                  confidence_interval_probability):
    np.random.seed(random_seed)
    samples = np.random.randn(1000)
    alpha = 1 - confidence_interval_probability
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    errors = []

    for subset_i in [30, 90, 200, 500, 750, 1000]:
        true_distr = norm(loc=0, scale=1 / np.sqrt(subset_i))
        expected_lb = true_distr.isf(lower_quantile)
        expected_ub = true_distr.isf(upper_quantile)

        samples_subset = samples[0:subset_i]
        EB = EmpiricalBootstrap(
            sample_generator=SampleGenerator(samples_subset)
        )
        rtv = EB.bootstrap_confidence_bounds(confidence_interval_probability,
                                             bootstrap_size)
        error = abs(rtv.lb_two_sided - expected_lb)
        error += abs(rtv.ub_two_sided - expected_ub)
        errors.append(error)

    assert np.all(np.diff(errors) < 0)


def test_transforming_sample_generator():
    transform = lambda x: 2*x[0] - x[1]
    data = [np.ones((5,)), 2*np.ones((5,))]
    EB = EmpiricalBootstrap(sample_generator=TransformingSampleGenerator(
        data=data, transform=transform
    ))
    new_data = EB.get_bootstrap_means(5)
    assert np.allclose(new_data, np.zeros(5,))


def test_random_state_changes():
    SG = SampleGenerator(data=[1,2,3])
    rs1 = str(SG._random_state.get_state())
    rs1_1 = str(SG._random_state.get_state())
    assert rs1 == rs1_1
    SG.new_bootstrap_sample()
    rs2 = str(SG._random_state.get_state())
    assert rs1 != rs2
