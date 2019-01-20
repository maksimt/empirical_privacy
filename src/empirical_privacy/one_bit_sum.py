from math import sqrt

import numpy as np
from scipy.stats import binom, norm

from experiment_framework.privacy_estimator_mixins import KNNFitterMixin
from experiment_framework.sampling_framework import GenSamples, GenSample, FitModel, \
    EvaluateStatisticalDistance, ComputeConvergenceCurve
from experiment_framework.asymptotic_analysis import ComputeAsymptoticAccuracy


def B_pmf(k, n, p):
    return binom(n, p).pmf(k)
def B0_pmf(k, n, p):
    return B_pmf(k, n-1, p)
def B1_pmf(k, n, p):
    return B_pmf(k-1, n-1, p)
def sd(N, P):
    return 0.5*np.sum(abs(B0_pmf(i, N, P) - B1_pmf(i, N, P)) for i in range(N+1))

class GenSampleOneBitSum(GenSample):

    def gen_sample(self, sample_number, dimension=3):
        seed_val = self.random_seed
        try:
            seed_val = 'seed{sv}doc{di}'.format(
                    di=self.dataset_settings['doc_ind'],
                    sv=seed_val
                )
        except KeyError:
            pass

        GenSample.set_simple_random_seed(sample_number, seed_val)

        n = self.dataset_settings['n_trials']
        p = self.dataset_settings['prob_success']

        num_samples = 1
        size = (num_samples, dimension)
        if self.dataset_settings['gen_distr_type'] == 'binom':
            B0 = binom.rvs(n - 1, p, size=1) + 0
            B1 = binom.rvs(n - 1, p, size=1) + 1
        elif self.dataset_settings['gen_distr_type'] == 'norm':
            sigma = sqrt((n - 0.75) / 12.0)
            mu = (n - 1.0) / 2
            B0 = norm.rvs(loc=mu + 0.25, scale=sigma, size=size)
            B1 = norm.rvs(loc=mu + 0.75, scale=sigma, size=size)
        elif self.dataset_settings['gen_distr_type'] == 'multidim_binom':
            if p < 0.5:
                p = 1 - p
            B0 = binom.rvs(1, 1-p, size=size)
            B1 = binom.rvs(1, p, size=size)
        else:
            raise ValueError('Unrecognized gen_distr_type={}'.format(
                    self.dataset_settings['gen_distr_type']))

        X0 = B0  # = np.concatenate((B0, B1))[:, np.newaxis]
        X1 = B1
        y0 = np.zeros((num_samples,))
        y1 = np.ones((num_samples,))

        if self.generate_positive_sample:
            return X1, y1
        else:
            return X0, y0


class GenSamplesOneBit(GenSamples(GenSampleOneBitSum,
                                  x_concatenator=np.vstack,
                                  generate_in_batch=True),
                       ):
    pass


class FitKNNModelOneBit(KNNFitterMixin(), FitModel(GenSamplesOneBit)):
    pass


class EvaluateKNNOneBitSD(EvaluateStatisticalDistance(
    samplegen=GenSamplesOneBit,
    model=FitKNNModelOneBit
    )):
    pass


class ComputeOneBitKNNConvergence(ComputeConvergenceCurve(EvaluateKNNOneBitSD)):
    pass

class OneBitAsymptoticAccuracy(
    ComputeAsymptoticAccuracy(ComputeOneBitKNNConvergence)):
    pass