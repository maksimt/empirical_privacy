import copy
import logging
from pprint import pformat

import luigi

from experiment_framework.asymptotic_analysis import \
    ComputeAsymptoticAccuracy
from experiment_framework.differential_privacy import ComputeBoundsForDelta
from experiment_framework.privacy_estimator_mixins import DensityEstFitterMixin, \
    ExpectationFitterMixin, KNNFitterMixin
from experiment_framework.utils.python_helpers import load_from
from experiment_framework.sampling_framework import GenSample, GenSamples, FitModel, \
    EvaluateStatisticalDistance
from experiment_framework.compute_convergence_curve import ComputeConvergenceCurve
from empirical_privacy.config import MIN_SAMPLES


class AllDeltas(luigi.WrapperTask):
    gen_sample_path = luigi.Parameter()
    dataset_settings = luigi.DictParameter()
    asymptotic_settings = luigi.DictParameter(default={})
    claimed_epsilon = luigi.FloatParameter()

    def requires(self):
        GS = load_from(self.gen_sample_path)
        # dict() so we can modify it without getting FrozenDict violations
        CLBDs = deltas_for_multiple_docs(
            dataset_settings=dict(self.dataset_settings),
            GS=GS,
            claimed_epsilon=self.claimed_epsilon,
            **self.asymptotic_settings)
        return list(CLBDs)


def deltas_for_multiple_docs(
        *args,
        claimed_epsilon: float,
        **kwargs
):
    AAs = asymptotics_for_multiple_docs(*args, **kwargs)

    class CLBDType(ComputeBoundsForDelta(type(AAs[0]))):
        pass

    for AA in AAs:
        yield CLBDType(claimed_epsilon=claimed_epsilon,
                       **AA.param_kwargs)


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
        dataset_settings: dict,
        GS: GenSample,
        gen_sample_kwargs={'generate_in_batch': True},
        fitter_kwargs={},
        fitter='knn',
        n_bootstraps=100,
        p=0.99,
        n_docs=10,
        confidence_interval_width=1,
        validation_set_size=64,
        n_max=256,
        min_samples=MIN_SAMPLES,
        in_memory=False,
        knn_curve_model='gyorfi',
        aa_factory_kwargs={}
):
    if 'doc_ind' in dataset_settings:
        logging.warning('doc_ind is overwritten; if you need granular control'
                        'consider building the AsymptoticAnalysis classes by '
                        'hand.')

    CCCType = build_convergence_curve_pipeline(GS, gen_sample_kwargs,
                                               fitter_kwargs, fitter)

    class AAType(ComputeAsymptoticAccuracy(CCCType, **aa_factory_kwargs)):
        pass

    AAs = []
    for doc_i in range(n_docs):
        ds = copy.deepcopy(dataset_settings)
        ds['doc_ind'] = doc_i
        AAs.append(AAType(
            n_bootstraps=n_bootstraps,
            confidence_interval_prob=p,
            confidence_interval_width=confidence_interval_width,
            n_max=n_max,
            min_samples=min_samples,
            dataset_settings=ds,
            validation_set_size=validation_set_size,
            in_memory=in_memory,
            knn_curve_model=knn_curve_model
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
            fitter_kwargs = {'statistic_column': 0}

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

    # we add the hash to the name because the fitter_kwargs are not stored by
    # luigi, and so it wouldn't recognize calls with different fitter_kwargs
    # otherwise
    FM.__name__ = gs_name + 'FitModel' + fitter \
                  + str(hash(pformat(fitter_kwargs)))

    class ESD(EvaluateStatisticalDistance(samplegen=GSs, model=FM)):
        pass

    ESD.__name__ = gs_name + 'EvaluateStatisticalDistance' + fitter

    class CCC(ComputeConvergenceCurve(ESD)):
        pass

    CCC.__name__ = gs_name + 'ComputeConvergenceCurve' + fitter

    return CCC


