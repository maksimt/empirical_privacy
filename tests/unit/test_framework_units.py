import pytest
import luigi
import pickle
import logging

from empirical_privacy import one_bit_sum_joblib
from empirical_privacy import one_bit_sum

@pytest.fixture()
def ds_rs():
    return {'dataset_settings': {'n_trials':40, 'prob_success':0.5,
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
        KNN = pickle.load(f)
    assert(hasattr(KNN, 'model'))

def test_compute_stat_dist(ds_rs):
    ESD = one_bit_sum.EvaluateKNNOneBitSD(training_set_size=200,
                                          validation_set_size=500,
                                          **ds_rs)
    luigi.build([ESD], local_scheduler=True, workers=1, log_level='ERROR')
    with ESD.output().open() as f:
        sd = pickle.load(f)['statistical_distance']
    assert(sd > 0 and sd < 0.3)

def test_compute_convergence_curve(ds_rs):
    CC = one_bit_sum.ComputeOneBitKNNConvergence(
        n_trials_per_training_set_size=3,
        n_max=33,
        n_steps=4,
        dataset_settings = ds_rs['dataset_settings'],
        validation_set_size = 10
    )
    luigi.build([CC], local_scheduler=True, workers=8, log_level='ERROR')
    with CC.output().open() as f:
        res = pickle.load(f)
    assert res['sd_matrix'].shape == (3,4)
    print(res['sd_matrix'])