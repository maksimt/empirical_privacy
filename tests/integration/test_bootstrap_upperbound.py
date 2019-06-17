import luigi
import dill
import pytest

from experiment_framework.helpers import AllAsymptotics
from empirical_privacy import one_bit_sum


@pytest.mark.parametrize(
    'neighbor_method', [
        'gyorfi',
        'sqrt_random_tiebreak',
        'sqrt'
    ])
def test_asymptotic_generator(neighbor_method):
    n = 7
    p = 0.5
    ds = {
        'n_trials': n,
        'prob_success': p,
        'gen_distr_type': 'multidim_binom',
    }
    asys = {
        'fitter'             : 'knn',
        'fitter_kwargs'      : {'neighbor_method': neighbor_method},
        'n_docs'                : 1,
        'n_trials_per_training_set_size': 5,
        'n_max'              : 2**9,
        'validation_set_size': 2**5
    }

    All = AllAsymptotics(
            gen_sample_path='empirical_privacy.one_bit_sum.GenSampleOneBitSum',
            dataset_settings=ds, asymptotic_settings=asys)
    luigi.build([All], local_scheduler=True, workers=1, log_level='ERROR')
    AA = All.requires()[0]
    with AA.output().open() as f:
        res = dill.load(f)
    assert 'upper_bound' in res