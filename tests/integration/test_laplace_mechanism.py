import pytest
import luigi
import dill
import numpy as np

from empirical_privacy.laplace_mechanism import GenSampleLaplaceMechanism
from experiment_framework.sampling_framework import GenSamples


@pytest.fixture()
def gs():
    gs = GenSampleLaplaceMechanism(
        dataset_settings={
            'dimension': 3,
            'epsilon'  : 0.1,
            'delta'    : 0
            },
        generate_positive_sample=True,
        random_seed='0'
        )
    return gs

def test_gen_sample_laplace_constructor(gs):
    X, y = gs.gen_sample(0)
    assert X.shape == (1,3)
    assert y ==1


@pytest.mark.parametrize('gen_positive_samples', [True, False])
def test_distinct_samples_generated(gen_positive_samples, gs):
    gen_samples_type = GenSamples(
        gen_sample_type=type(gs),
        x_concatenator='numpy.vstack',
        generate_in_batch=True
    )
    gen_samples_object = gen_samples_type(
        num_samples=8,
        generate_positive_samples=gen_positive_samples,
        dataset_settings={
            'dimension': 3,
            'epsilon'  : 0.1,
            'delta'    : 0
            },
        random_seed='0'
    )
    luigi.build([gen_samples_object], local_scheduler=True, log_level='ERROR')
    with gen_samples_object.output().open() as f:
        output = dill.load(f)

    # assert all rows are different row-wise
    assert np.all(np.abs(np.diff(output['X'], axis=0)) > 0)
