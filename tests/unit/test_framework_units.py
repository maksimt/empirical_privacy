import pytest
import luigi
import pickle
import logging

from empirical_privacy import one_bit_sum_joblib
from empirical_privacy import one_bit_sum

@pytest.fixture()
def ds_rs():
    return {'dataset_settings': {'n_trials':100, 'prob_success':0.3,
                                    'gen_distr_type':'binom'},
            'random_seed':'1338'}

def test_pytest():
    assert 3 == (2+1)

def test_import():
    X0, X1, y0, y1 = one_bit_sum_joblib.gen_data(10, 0.5, 100, 0)
    assert X0.shape[0] == 100

def test_gen_samples_one_bit(ds_rs):
    GSTask = one_bit_sum.GenSamplesOneBit(
        generate_positive_samples = True,
        num_samples = 23,
        **ds_rs
    )
    luigi.build([GSTask], local_scheduler=True, workers=1, log_level='ERROR')
    with GSTask.output().open() as f:
        samples = pickle.load(f)
    assert(samples['X'].size==23)

def test_fit_model_one_bit(ds_rs):
    FMTask = one_bit_sum.FitKNNModelOneBit(samples_per_class=11, **ds_rs)
    luigi.build([FMTask], local_scheduler=True, workers=1, log_level='ERROR')
    with FMTask.output().open() as f:
        model = pickle.load(f)
    assert(hasattr(model, 'fit'))