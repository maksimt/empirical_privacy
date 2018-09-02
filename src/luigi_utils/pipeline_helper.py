

from luigi_utils.sampling_framework import GenSample, GenSamples, FitModel, \
    EvaluateStatisticalDistance, ComputeConvergenceCurve
from luigi_utils.privacy_estimator_mixins import DensityEstFitterMixin, \
    ExpectationFitterMixin, KNNFitterMixin


def build_convergence_curve_pipeline(GenSampleType: GenSample,
                                     generate_in_batch=False,
                                     fitter = 'knn',
                                     ) -> ComputeConvergenceCurve:
    gs_name = GenSampleType.__name__
    GSs = type(gs_name+'GenSamples',
               (GenSamples(GenSampleType, generate_in_batch=generate_in_batch),)
               , {}
               )

    if fitter=='knn':
        F = KNNFitterMixin
    elif fitter=='density':
        F = DensityEstFitterMixin
    elif fitter=='expectation':
        F = ExpectationFitterMixin

    FM = type(gs_name+'FitModel'+fitter,
              (F(), FitModel(GSs)), {}
              )

    ESD = type(gs_name+'EvaluateStatisticalDistance'+fitter,
               (EvaluateStatisticalDistance(samplegen=GSs, model=FM),), {}
               )

    CCC = type(gs_name+'ComputeConvergenceCurve'+fitter,
               (ComputeConvergenceCurve(ESD),), {}
               )

    return CCC

def build_convergence_curve_pipeline2(GenSampleType: GenSample,
                                     generate_in_batch=False,
                                     fitter = 'knn',
                                     ) -> ComputeConvergenceCurve:
    gs_name = GenSampleType.__name__

    class GSs(GenSamples(GenSampleType, generate_in_batch=True)):
        pass
    GSs.__name__ = gs_name+'GenSamples'

    if fitter=='knn':
        F = KNNFitterMixin
    elif fitter=='density':
        F = DensityEstFitterMixin
    elif fitter=='expectation':
        F = ExpectationFitterMixin

    class FM(F(), FitModel(GSs)):
        pass
    FM.__name__ = gs_name+'FitModel'+fitter

    class ESD(EvaluateStatisticalDistance(samplegen=GSs, model=FM)):
        pass
    ESD.__name__ = gs_name+'EvaluateStatisticalDistance'+fitter

    class CCC(ComputeConvergenceCurve(ESD)):
        pass
    CCC.__name__ = gs_name+'ComputeConvergenceCurve'+fitter

    return CCC