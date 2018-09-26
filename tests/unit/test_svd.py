import pytest

import dill
import luigi

from empirical_privacy.row_distributed_common import \
    gen_attacker_and_defender_indices
from empirical_privacy.row_distributed_svd import CCCSVD, GenSVDSample, CCCFVSVD

def test_gen_indices():
    doc_ind = 7
    n_adv_with = 0
    for i in range(1000):
        IDX = gen_attacker_and_defender_indices(10, 0.4, doc_ind=doc_ind,
                                                seed=str(i))
        I1 = IDX['I_defender_with']
        I0 = IDX['I_defender_without']
        if doc_ind in IDX['I_attacker']:
            n_adv_with += 1
        assert doc_ind in I1 and doc_ind not in I0
    assert 300 <= n_adv_with <= 500


@pytest.fixture(scope='function')
def ccc_kwargs(request):
    ds = {
        'dataset_name' : 'PCR_Test',
        'part_fraction': 0.3,
        'doc_ind'      : 33,
        'SVD_type'     : request.param,
        'SVD_k'        : 5
        }
    return {
        'n_trials_per_training_set_size': 3,
        'n_max'                         : 16,
        'dataset_settings'              : ds,
        'validation_set_size'           : 8
        }

@pytest.mark.parametrize('ccc_kwargs',
                         [
                             'hidden_eigs',
                             'exposed_eigs',
                             'full_correlation'
                        ],
                         indirect=['ccc_kwargs'])
def test_CCC(ccc_kwargs):
    CCCSVD_obj = CCCSVD(**ccc_kwargs)
    luigi.build([CCCSVD_obj], local_scheduler=True, workers=8,
                log_level='ERROR')
    with CCCSVD_obj.output().open() as f:
        res = dill.load(f)
    assert res['sd_matrix'].shape == (3,2)

@pytest.mark.parametrize('ccc_kwargs', ['hidden_eigs'], indirect=['ccc_kwargs'])
def test_GS_20ng(ccc_kwargs):
    ds = ccc_kwargs['dataset_settings']
    ds['dataset_name'] = 'ml-1m'

    GS = GenSVDSample(dataset_settings=ds,
                      random_seed='0',
                      generate_positive_sample=True,
                      sample_number=0)
    luigi.build([GS], local_scheduler=True, workers=1,
                log_level='ERROR')
    with GS.output().open() as f:
        res = dill.load(f)
    assert res.x.size >= 6

@pytest.mark.parametrize('ccc_kwargs', ['hidden_eigs'], indirect=['ccc_kwargs'])
def test_full_view_samples(ccc_kwargs):
    CCCSVD_obj = CCCFVSVD(**ccc_kwargs)
    luigi.build([CCCSVD_obj], local_scheduler=True, workers=8,
                log_level='ERROR')
    with CCCSVD_obj.output().open() as f:
        res = dill.load(f)
    assert res['sd_matrix'].shape == (3, 2)