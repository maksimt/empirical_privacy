from typing import Sequence
import dill
import collections
import copy

import pandas as pd

from luigi_utils.privacy_estimator_mixins import DensityEstFitterMixin, \
    ExpectationFitterMixin, KNNFitterMixin
from luigi_utils.sampling_framework import GenSample, GenSamples, FitModel, \
    EvaluateStatisticalDistance, ComputeConvergenceCurve, _ComputeConvergenceCurve


def build_convergence_curve_pipeline(GenSampleType: GenSample,
                                     generate_in_batch=False,
                                     fitter='knn',
                                     ) -> ComputeConvergenceCurve:
    gs_name = GenSampleType.__name__

    class GSs(GenSamples(GenSampleType,
                         generate_in_batch=generate_in_batch)):
        pass

    GSs.__name__ = gs_name + 'GenSamples'

    if fitter == 'knn':
        F = KNNFitterMixin
    elif fitter == 'density':
        F = DensityEstFitterMixin
    elif fitter == 'expectation':
        F = ExpectationFitterMixin

    class FM(F(), FitModel(GSs)):
        pass

    FM.__name__ = gs_name + 'FitModel' + fitter

    class ESD(EvaluateStatisticalDistance(samplegen=GSs, model=FM)):
        pass

    ESD.__name__ = gs_name + 'EvaluateStatisticalDistance' + fitter

    class CCC(ComputeConvergenceCurve(ESD)):
        pass

    CCC.__name__ = gs_name + 'ComputeConvergenceCurve' + fitter

    return CCC


def _flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.Mapping):
            items.extend(_flatten_dict(v, None, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_completed_CCCs_into_dataframe(
        CCCs: Sequence[_ComputeConvergenceCurve]
    ):
    res = []
    for CCC in CCCs:
        if CCC.complete():
            with CCC.output().open() as f:
                dat = dill.load(f)
            as_dict = _flatten_dict(CCC.param_kwargs)
            S = dat['sd_matrix']
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