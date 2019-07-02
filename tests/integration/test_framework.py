import copy

import dill
import luigi
import numpy as np
import pytest
import time

from empirical_privacy import one_bit_sum
from empirical_privacy.config import MIN_SAMPLES, SAMPLES_BASE
from experiment_framework.utils.helpers import build_convergence_curve_pipeline, \
    AllAsymptotics
from notebook_context.pandas_interface import load_completed_CCCs_into_dataframe
from experiment_framework.utils.python_helpers import load_from
from experiment_framework.sampling_framework import GenSamples


@pytest.fixture(scope='session')
def ds_rs():
    n = 40
    p = 0.5
    return {
        'dataset_settings': {
            'n_trials'      : n,
            'prob_success'  : p,
            'gen_distr_type': 'binom',
        },
        'random_seed'     : '1338',
        'sd'              : one_bit_sum.sd(n, p)
    }


def test_gen_samples_one_bit(ds_rs):
    try:
        ds_rs.pop('sd')
    except KeyError:
        pass
    GSTask = one_bit_sum.GenSamplesOneBit(
        generate_positive_samples=True,
        num_samples=23,
        **ds_rs
    )
    luigi.build([GSTask], local_scheduler=True, workers=1, log_level='ERROR')
    with GSTask.output().open() as f:
        samples = dill.load(f)
    assert (samples['X'].size == 23)


def test_fit_model_one_bit(ds_rs):
    FMTask = one_bit_sum.FitKNNModelOneBit(samples_per_class=11, **ds_rs)
    luigi.build([FMTask], local_scheduler=True, workers=1, log_level='ERROR')
    with FMTask.output().open() as f:
        model = dill.load(f)
    assert ('KNN' in model)


def test_compute_stat_dist(ds_rs):
    try:
        ds_rs.pop('sd')
    except KeyError:
        pass
    ESD = one_bit_sum.EvaluateKNNOneBitSD(training_set_size=1000,
                                          validation_set_size=500,
                                          **ds_rs)
    luigi.build([ESD], local_scheduler=True, workers=1, log_level='INFO')
    with ESD.output().open() as f:
        sd = dill.load(f)['statistical_distance']
    assert (sd > 0 and sd < 1)


@pytest.fixture(scope='session')
def ccc_kwargs(ds_rs):
    return {
        'n_trials_per_training_set_size': 3,
        'n_max'                         : 2 ** 9,
        'dataset_settings'              : ds_rs['dataset_settings'],
        'validation_set_size'           : 10
    }


@pytest.fixture(scope='session')
def simple_ccc(ccc_kwargs):
    CC = one_bit_sum.ComputeOneBitKNNConvergence(**ccc_kwargs)
    luigi.build([CC], local_scheduler=True, workers=8, log_level='ERROR')
    with CC.output().open() as f:
        res = dill.load(f)
    return res


@pytest.fixture(scope='session')
def expected_accuracy_matrix_shape(ccc_kwargs):
    n_row = ccc_kwargs['n_trials_per_training_set_size']
    pow_min = np.floor(np.log(MIN_SAMPLES) / np.log(SAMPLES_BASE)
                       + np.spacing(1)).astype(np.int)
    pow_max = np.floor(np.log(ccc_kwargs['n_max']) / np.log(SAMPLES_BASE)
                       + np.spacing(1)).astype(np.int)
    n_steps = pow_max - pow_min + 1
    n_col = n_steps
    return (n_row, n_col)


def test_compute_convergence_curve(simple_ccc, expected_accuracy_matrix_shape,
                                   ccc_kwargs):
    assert simple_ccc['training_set_sizes'][0] == MIN_SAMPLES
    assert simple_ccc['training_set_sizes'][-1] == ccc_kwargs['n_max']
    assert simple_ccc['accuracy_matrix'].shape == expected_accuracy_matrix_shape


@pytest.fixture(scope='session')
def built_ccc(ccc_kwargs):
    CCC = build_convergence_curve_pipeline(one_bit_sum.GenSampleOneBitSum,
                                           gensample_kwargs={
                                               'generate_in_batch': True
                                           })
    CCC2_inst = CCC(**ccc_kwargs)
    start_clock = time.clock()
    luigi.build([CCC2_inst], local_scheduler=True, workers=8, log_level='ERROR')
    cputime = time.clock() - start_clock
    with CCC2_inst.output().open() as f:
        res = dill.load(f)
    return {'res': res, 'cputime': cputime}


def test_ccc_pipeline_builder(simple_ccc, built_ccc):
    assert np.allclose(built_ccc['res']['accuracy_matrix'], simple_ccc['accuracy_matrix'])


def test_built_ccc_cached_correctly(built_ccc, ccc_kwargs):
    AbraCadabra = build_convergence_curve_pipeline(
        one_bit_sum.GenSampleOneBitSum,
        gensample_kwargs={'generate_in_batch': True})
    AbraCadabra_inst = AbraCadabra(**ccc_kwargs)
    start_clock = time.clock()
    luigi.build([AbraCadabra_inst], local_scheduler=True, workers=8,
                log_level='ERROR')
    cputime = time.clock() - start_clock
    assert cputime < 1 / 5.0 * built_ccc['cputime']


def test_delete_deps(built_ccc, ccc_kwargs):
    AbraCadabra = build_convergence_curve_pipeline(
        one_bit_sum.GenSampleOneBitSum,
        gensample_kwargs={'generate_in_batch': True})
    AbraCadabra_inst = AbraCadabra(**ccc_kwargs)
    n_del = AbraCadabra_inst.delete_deps()
    assert n_del >= 10
    start_clock = time.clock()
    luigi.build([AbraCadabra_inst], local_scheduler=True, workers=8,
                log_level='ERROR')
    cputime = time.clock() - start_clock
    assert 0.5 * built_ccc['cputime'] < cputime < 1.5 * built_ccc['cputime']


@pytest.mark.parametrize('fitter', ['density', 'expectation'])
def test_other_ccc_fitters(fitter, ccc_kwargs, expected_accuracy_matrix_shape):
    CCC = build_convergence_curve_pipeline(one_bit_sum.GenSampleOneBitSum,
                                           gensample_kwargs={
                                               'generate_in_batch': True
                                           },
                                           fitter=fitter)
    TheCCC = CCC(**ccc_kwargs)
    luigi.build([TheCCC], local_scheduler=True, workers=1,
                log_level='ERROR')
    with TheCCC.output().open() as f:
        res = dill.load(f)
    assert res['accuracy_matrix'].shape == expected_accuracy_matrix_shape
    assert np.all((0 <= res['accuracy_matrix']) & (res['accuracy_matrix'] <= 1))


def test_load_CCCs_into_DF(ccc_kwargs, expected_accuracy_matrix_shape):
    CCC = build_convergence_curve_pipeline(one_bit_sum.GenSampleOneBitSum,
                                           gensample_kwargs={
                                               'generate_in_batch': True
                                           },
                                           fitter='knn')
    CCCs = []
    for rs in range(5, 10):
        ck = copy.deepcopy(ccc_kwargs)
        ck['dataset_settings']['n_trials'] = rs
        CCCs.append(CCC(**ck))
    luigi.build(CCCs, local_scheduler=True, workers=4, log_level='ERROR')
    DF = load_completed_CCCs_into_dataframe(CCCs)
    n_rows_exp = np.prod(expected_accuracy_matrix_shape) * 5
    assert DF.shape == (n_rows_exp, 10)


def test_asymptotic_generator(ds_rs):
    All = AllAsymptotics(
        gen_sample_path='empirical_privacy.one_bit_sum.GenSampleOneBitSum',
        dataset_settings=ds_rs['dataset_settings'],
        asymptotic_settings={'n_docs': 1,
                             'n_max': 2 ** 9,
                             'knn_curve_model': 'sqrt'
        },

    )
    luigi.build([All], local_scheduler=True, workers=8, log_level='ERROR')
    AA = All.requires()[0]
    with AA.output().open() as f:
        res = dill.load(f)
    assert 'upper_bound' in res


def test_deterministic(ds_rs):
    kwargs = dict(gen_sample_path='empirical_privacy.one_bit_sum.GenSampleOneBitSum',
                  dataset_settings=ds_rs['dataset_settings'],
                  asymptotic_settings={
                      'n_docs'         : 1,
                      'n_max'          : 2 ** 7,
                      'knn_curve_model': 'sqrt'
                  })
    All = AllAsymptotics(**kwargs)
    luigi.build([All], local_scheduler=True, workers=4, log_level='ERROR')
    AA = All.requires()[0]
    with AA.output().open() as f:
        res1 = dill.load(f)
    AA.delete_deps()
    AA.delete_outputs()

    All = AllAsymptotics(**kwargs)
    luigi.build([All], local_scheduler=True, workers=4, log_level='ERROR')
    AA = All.requires()[0]
    with AA.output().open() as f:
        res2 = dill.load(f)
    AA.delete_deps()
    AA.delete_outputs()

    assert res1 == res2


def test_importlib(ds_rs):
    _path = 'empirical_privacy.one_bit_sum.GenSampleOneBitSum'
    GS = load_from(_path)
    try:
        ds_rs.pop('sd')
    except KeyError:
        pass
    GSTask = GS(
        generate_positive_sample=True,
        sample_number=0,
        **ds_rs
    )
    luigi.build([GSTask], local_scheduler=True, workers=1, log_level='ERROR')
    with GSTask.output().open() as f:
        samples = dill.load(f)
    assert samples.y[0] == 1


def test_attr_string_handling(ds_rs):
    try:
        ds_rs.pop('sd')
    except KeyError:
        pass
    GSs = GenSamples(one_bit_sum.GenSampleOneBitSum,
                     x_concatenator='numpy.concatenate')
    assert hasattr(GSs.x_concatenator, '__call__')
