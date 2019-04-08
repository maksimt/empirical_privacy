import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix

from experiment_framework.sampling_framework import GenSample
from dataset_utils.common import load_dataset


class BrokenSparseVector(GenSample):

    @property
    def X(self) -> csr_matrix:
        if not hasattr(self, 'Xtr'):
            ds = load_dataset(self.dataset_settings['dataset_name'])
            self.Xtr = ds['Xtr'].astype(np.double)

    def gen_sample(self, sample_number, dimension=3):
        c = self.dataset_settings['c']
        T = self.dataset_settings['T']