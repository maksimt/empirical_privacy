import copy
from typing import Sequence

import dill
import pandas as pd

from experiment_framework.asymptotic_analysis import _ComputeAsymptoticAccuracy
from experiment_framework.utils.python_helpers import _flatten_dict
from experiment_framework.sampling_framework import _ComputeConvergenceCurve


def load_completed_CCCs_into_dataframe(
        CCCs: Sequence[_ComputeConvergenceCurve]
):
    res = []
    for CCC in CCCs:
        if CCC.complete():
            with CCC.output().open() as f:
                dat = dill.load(f)
            as_dict = _flatten_dict(CCC.param_kwargs)
            S = dat['accuracy_matrix']
            tss = dat['training_set_sizes']
            (ntri, nsamp) = S.shape
            for tri in range(ntri):
                for samp_i in range(nsamp):
                    rtv_dict = copy.deepcopy(as_dict)
                    rtv_dict['trial'] = tri
                    rtv_dict['training_set_size'] = tss[samp_i]
                    rtv_dict['classifier_accuracy'] = S[tri, samp_i]
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