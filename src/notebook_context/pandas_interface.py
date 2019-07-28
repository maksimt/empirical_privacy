import copy
from typing import Sequence

import dill
import pandas as pd

from experiment_framework.asymptotic_analysis import _ComputeAsymptoticAccuracy
from experiment_framework.utils.python_helpers import _flatten_dict
from experiment_framework.compute_convergence_curve import _ComputeConvergenceCurve


def CCCS_for_AA(AA: _ComputeAsymptoticAccuracy):
    assert AA.complete(), 'CCCs only known when an AA is complete'
    with AA.output().open() as f:
        res = dill.load(f)
    return [AA.CCC_job(i) for i in range(res['n_curves_done'])]


def load_completed_CCCs_into_dataframe(
        CCCs: Sequence[_ComputeConvergenceCurve]
):
    res = []
    for CCC in CCCs:
        if CCC.complete():
            with CCC.output().open() as f:
                dat = dill.load(f)
            as_dict = _flatten_dict(CCC.param_kwargs)
            tss_accuracy = dat['training_set_size_to_accuracy']
            for tss, accuracies in tss_accuracy.items():
                for (tri, accuracy) in enumerate(accuracies):
                    rtv_dict = copy.deepcopy(as_dict)
                    rtv_dict['trial'] = tri
                    rtv_dict['training_set_size'] = tss
                    rtv_dict['classifier_accuracy'] = accuracy
                    res.append(rtv_dict)
    DF = pd.DataFrame.from_dict(res)
    return DF


def load_completed_AAs_into_dataframe(
        AAs: Sequence[_ComputeAsymptoticAccuracy]
):
    res = []
    for AA in AAs:
        if not AA.complete():
            continue
        with AA.output().open() as f:
            dat = dill.load(f)
        as_dict = _flatten_dict(AA.param_kwargs)
        dat.update(as_dict)
        res.append(dat)
    DF = pd.DataFrame.from_dict(res)
    return DF