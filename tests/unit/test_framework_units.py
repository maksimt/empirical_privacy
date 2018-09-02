import pytest
import luigi
import pickle
import logging
import numpy as np

from empirical_privacy import one_bit_sum_joblib
from empirical_privacy import one_bit_sum
from luigi_utils.pipeline_helper import build_convergence_curve_pipeline, \
    build_convergence_curve_pipeline2

@pytest.fixture(scope='session')
def ds_rs():
    return {'dataset_settings': {'n_trials':4000, 'prob_success':0.5,
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
    assert(sd > 0 and sd < 1)


@pytest.fixture(scope='session')
def ccc_kwargs(ds_rs):
    return {
        'n_trials_per_training_set_size': 3,
        'n_max': 30,
        'n_steps': 4,
        'dataset_settings': ds_rs['dataset_settings'],
        'validation_set_size': 10
    }

@pytest.fixture(scope='session')
def simple_ccc(ccc_kwargs):
    CC = one_bit_sum.ComputeOneBitKNNConvergence(**ccc_kwargs)
    luigi.build([CC], local_scheduler=True, workers=8, log_level='ERROR')
    with CC.output().open() as f:
        res = pickle.load(f)
    return res

def test_compute_convergence_curve(simple_ccc):
    assert simple_ccc['sd_matrix'].shape == (3,4)

def test_ccc_pipeline_builder( simple_ccc, ccc_kwargs):
    CCC = build_convergence_curve_pipeline2(one_bit_sum.GenSampleOneBitSum)
    CCC2_inst = CCC(**ccc_kwargs)
    luigi.build([CCC2_inst], local_scheduler=True, workers=8, log_level='ERROR')
    with CCC2_inst.output().open() as f:
        res = pickle.load(f)
    print(res['sd_matrix'])
    print(simple_ccc['sd_matrix'])
    assert np.allclose(res['sd_matrix'], simple_ccc['sd_matrix'])