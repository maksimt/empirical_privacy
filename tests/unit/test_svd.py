import pytest
from empirical_privacy.row_distributed_common import \
    gen_attacker_and_defender_indices


def test_gen_indices():
    doc_ind = 7
    n_adv_with = 0
    for i in range(1000):
        IDX = gen_attacker_and_defender_indices(10,0.4,doc_ind=doc_ind,
                                                seed=str(i))
        I1 = IDX['I_defender_with']
        I0 = IDX['I_defender_without']
        if doc_ind in IDX['I_attacker']:
            n_adv_with += 1
        assert doc_ind in I1 and doc_ind not in I0
    assert 300 <= n_adv_with <= 500