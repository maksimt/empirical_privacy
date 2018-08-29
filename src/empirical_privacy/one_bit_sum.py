import numpy as np
from scipy.stats import binom, norm
from math import sqrt

from luigi_utils.sampling_framework import GenSamples, GenSample, FitModel,\
    KNNFitterMixin, EvaluateStatisticalDistance

class GenSampleOneBitSum(GenSample):
    def gen_sample(self, dataset_settings, generate_positive_sample,
                   sample_number: int, random_seed: str):

        GenSample.set_simple_random_seed(sample_number, random_seed)

        n = dataset_settings['n_trials']
        p = dataset_settings['prob_success']

        num_samples = 1
        if dataset_settings['gen_distr_type'] == 'binom':
            B0 = binom.rvs(n - 1, p, size=num_samples) + 0
            B1 = binom.rvs(n - 1, p, size=num_samples) + 1
        elif dataset_settings['gen_distr_type'] == 'norm':
            sigma = sqrt((n - 0.75) / 12.0)
            mu = (n - 1.0) / 2
            B0 = norm.rvs(loc=mu + 0.25, scale=sigma, size=num_samples)
            B1 = norm.rvs(loc=mu + 0.75, scale=sigma, size=num_samples)

        X0 = B0  # = np.concatenate((B0, B1))[:, np.newaxis]
        X1 = B1
        y0 = np.zeros((num_samples,))
        y1 = np.ones((num_samples,))

        if generate_positive_sample:
            return X1, y1
        else:
            return X0, y0

class GenSamplesOneBit(GenSamples(GenSampleOneBitSum)):
    pass

class FitKNNModelOneBit(KNNFitterMixin(), FitModel(GenSamplesOneBit)):
    pass

class EvaluateKNNOneBitSD(EvaluateStatisticalDistance(
    samplegen=GenSamplesOneBit,
    model=FitKNNModelOneBit
)):
    pass