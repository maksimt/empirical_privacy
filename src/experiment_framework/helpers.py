from typing import Sequence
import dill
import copy
import logging

import luigi
import pandas as pd

from experiment_framework.privacy_estimator_mixins import DensityEstFitterMixin, \
    ExpectationFitterMixin, KNNFitterMixin
from experiment_framework.python_helpers import load_from, _flatten_dict
from experiment_framework.sampling_framework import GenSample, GenSamples, FitModel, \
    EvaluateStatisticalDistance, ComputeConvergenceCurve, \
    _ComputeConvergenceCurve
from experiment_framework.asymptotic_analysis import \
    ComputeAsymptoticAccuracy, _ComputeAsymptoticAccuracy


class AllAsymptotics(luigi.WrapperTask):
    gen_sample_path = luigi.Parameter()
    dataset_settings = luigi.DictParameter()
    asymptotic_settings = luigi.DictParameter(default={})

    def requires(self):
        GS = load_from(self.gen_sample_path)
        # dict() so we can modify it without getting FrozenDict violations
        AAs = asymptotics_for_multiple_docs(
            dict(self.dataset_settings),
            GS,
            **self.asymptotic_settings
        )
        return AAs


def asymptotics_for_multiple_docs(
        dataset_settings : dict,
        GS : GenSample,
        gen_sample_kwargs = {'generate_in_batch': True},
        fitter_kwargs = {},
        fitter = 'knn',
        t=0.01,
        p=0.99,
        n_docs=10,
        n_trials_per_training_set_size=10,
        validation_set_size=64,
        n_max=256,
    ):
    if 'doc_ind' in dataset_settings:
        logging.warning('doc_ind is overwritten; if you need granular control'
                        'consider building the AsymptoticAnalysis classes by '
                        'hand.')

    CCCType = build_convergence_curve_pipeline(GS, gen_sample_kwargs,
                                               fitter_kwargs, fitter)

    class AAType(ComputeAsymptoticAccuracy(CCCType)):
        pass

    AAs = []
    for doc_i in range(n_docs):
        ds = copy.deepcopy(dataset_settings)
        ds['doc_ind'] = doc_i
        AAs.append(AAType(
            confidence_interval_width=t,
            confidence_interval_prob=p,
            n_trials_per_training_set_size=n_trials_per_training_set_size,
            n_max=n_max,
            dataset_settings=ds,
            validation_set_size=validation_set_size
            )
        )
    return AAs


def build_convergence_curve_pipeline(GenSampleType: GenSample,
                                     gensample_kwargs,
                                     fitter_kwargs=None,
                                     fitter='knn',
                                     ) -> ComputeConvergenceCurve:
    gs_name = GenSampleType.__name__

    if fitter_kwargs is None:
        fitter_kwargs = {}
        if fitter is not 'knn':
            fitter_kwargs = {'statistic_column':0}

    class GSs(GenSamples(GenSampleType, **gensample_kwargs)):
        pass

    GSs.__name__ = gs_name + 'GenSamples'

    if fitter == 'knn':
        F = KNNFitterMixin
    elif fitter == 'density':
        F = DensityEstFitterMixin
    elif fitter == 'expectation':
        F = ExpectationFitterMixin

    class FM(F(**fitter_kwargs), FitModel(GSs)):
        pass

    FM.__name__ = gs_name + 'FitModel' + fitter

    class ESD(EvaluateStatisticalDistance(samplegen=GSs, model=FM)):
        pass

    ESD.__name__ = gs_name + 'EvaluateStatisticalDistance' + fitter

    class CCC(ComputeConvergenceCurve(ESD)):
        pass

    CCC.__name__ = gs_name + 'ComputeConvergenceCurve' + fitter

    return CCC


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