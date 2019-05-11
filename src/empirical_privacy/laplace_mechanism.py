from math import exp

import numpy as np

from experiment_framework.sampling_framework import (GenSample, GenSamples,
     FitModel, EvaluateStatisticalDistance)
from experiment_framework.privacy_estimator_mixins import KNNFitterMixin


class GenSampleLaplaceMechanism(GenSample):
    """
    Generate a sample of the form (X=database_0 + noise, y=0) or (
    X=database_1, y=1). Noise is distributed as Laplace(0, sensitivity/epsilon).
    Also, if y=1 with probability (1/exp(epsilon)) we generate a sample that
    can only be generated by the y=1 distribution, and can easily be
    classified.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.X0 = self.dataset_settings['database_0']
        self.X1 = self.dataset_settings['database_1']
        if not isinstance(self.X0, np.ndarray):
            self.X0 = np.array(self.X0)
        if not isinstance(self.X1, np.ndarray):
            self.X1 = np.array(self.X1)
        self.X0 = np.reshape(self.X0, (1, self.X0.size))
        self.X1 = np.reshape(self.X1, (1, self.X1.size))

        self.sensitivity = self.dataset_settings['sensitivity']
        self.one = np.ones((1,))
        self.zero = np.zeros((1,))
        self.epsilon = self.dataset_settings['epsilon']
        self.claimed_epsilon = self.dataset_settings['claimed_epsilon']

        # samples are generated according to the actual epsilon
        # but the experiment calls for natural samples to be drawn with
        # probability reciprocal to the claimed epsilon rather than the
        # actual epsilon
        self.laplace_scale = self.sensitivity / self.epsilon
        self.probability_of_natural_sample = 1 / (exp(self.claimed_epsilon))
        self.probability_of_alternative_sample = 1 - self.probability_of_natural_sample
        # we output an alternate sample that has negligible probability
        # (approx exp(-1000)) of being generated by the laplace distribution
        self.alternative_sample_noise = (-1000 * self.laplace_scale
                                         * np.ones_like(self.X1))

    def gen_sample(self, sample_number):
        seed_val = self.random_seed
        try:
            seed_val = 'seed{sv}doc{di}'.format(
                di=self.dataset_settings['doc_ind'],
                sv=seed_val
                )
        except KeyError:
            pass

        GenSample.set_simple_random_seed(sample_number, seed_val)

        if self.generate_positive_sample:
            X = self.X1
            y = self.one
        else:
            X = self.X0
            y = self.zero

        Xh = X + np.random.laplace(loc=0,  # mu
                                   scale=self.laplace_scale,
                                   size=X.size)
        if (self.generate_positive_sample and
                np.random.rand() <= self.probability_of_alternative_sample):
            Xh = X + self.alternative_sample_noise

        return Xh, y

class GenSamplesLaplace(GenSamples(GenSampleLaplaceMechanism,
                                   x_concatenator=np.vstack,
                                   generate_in_batch=True)):
    pass

class FitKNNModelLaplace(KNNFitterMixin(), FitModel(GenSamplesLaplace)):
    pass

class EvaluateKNNLaplaceStatDist(EvaluateStatisticalDistance(
    samplegen=GenSamplesLaplace,
    model=FitKNNModelLaplace
    )):
    pass