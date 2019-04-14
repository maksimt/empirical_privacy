import numpy as np

from experiment_framework.sampling_framework import GenSample


class GenSampleLaplaceMechanism(GenSample):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X0 = np.zeros((1, self.dataset_settings['dimension']))
        self.X1 = np.zeros((1, self.dataset_settings['dimension']))
        self.X1[0,0] = 1
        self.sensitivity = 1.
        self.one = np.ones((1,))
        self.zero = np.zeros((1,))

        self.epsilon = self.dataset_settings['epsilon']


    def gen_sample(self, sample_number):
        np.random.seed(sample_number)

        if self.generate_positive_sample:
            X = self.X1
            y = self.one
        else:
            X = self.X0
            y = self.zero

        Xh = X + np.random.laplace(loc=0,  # mu
                                   scale=self.sensitivity / self.epsilon,
                                   size=X.size)

        Xh = np.round(Xh, 6)  # mitigate LSB attack

        return Xh, y
