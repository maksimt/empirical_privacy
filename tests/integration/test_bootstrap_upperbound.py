import luigi
import dill

from experiment_framework.helpers import AllAsymptotics
from empirical_privacy import one_bit_sum


def test_asymptotic_generator():
    n = 7
    p = 0.5
    ds = {
        'n_trials': n, 'prob_success': p, 'gen_distr_type': 'binom',
    }
    asys = {
        'fitter'             : 'knn',
        'fitter_kwargs'      : {'neighbor_method': 'gyorfi'},
        'n_docs'                : 1,
        'n_trials_per_training_set_size': 10,
        'n_max'              : 2**10,
        'validation_set_size': 2**8
    }

    All = AllAsymptotics(
            gen_sample_path='empirical_privacy.one_bit_sum.GenSampleOneBitSum',
            dataset_settings=ds, asymptotic_settings=asys)
    luigi.build([All], local_scheduler=True, workers=1, log_level='ERROR')
    AA = All.requires()[0]
    with AA.output().open() as f:
        res = dill.load(f)
    assert 'mean' in res and 'upper_bound' in res