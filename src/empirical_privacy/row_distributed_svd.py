from copy import deepcopy
from functools import partial

import numpy as np
from scipy.sparse import csr_matrix, linalg, vstack as sparse_vstack, issparse

from dataset_utils.common import load_dataset
from empirical_privacy.row_distributed_common import \
    gen_attacker_and_defender_indices
from luigi_utils.privacy_estimator_mixins import KNNFitterMixin
from luigi_utils.sampling_framework import GenSample, GenSamples, FitModel, \
    EvaluateStatisticalDistance, ComputeConvergenceCurve


def svd_dataset_settings(part_fraction=0.3,
                         dataset_name='ml-1m',
                         SVD_type='hidden_eigs',
                         SVD_k=14):
    return {
        'part_fraction': part_fraction,
        'dataset_name' : dataset_name,
        'SVD_type'     : SVD_type,
        'SVD_k'        : SVD_k
        }


def gen_SVD_CCCs_for_multiple_docs(n_docs=10,
                                   n_trials_per_training_set_size=3,
                                   validation_set_size=64,
                                   n_max=256,
                                   dataset_settings=None
                                   ):
    if dataset_settings is None:
        dataset_settings = svd_dataset_settings()

    CCCs = []
    for doc_i in range(n_docs):
        ds = deepcopy(dataset_settings)
        ds['doc_ind'] = doc_i
        CCCs.append(CCCSVD(
            n_trials_per_training_set_size=n_trials_per_training_set_size,
            n_max=n_max,
            dataset_settings=ds,
            validation_set_size=validation_set_size
            )
            )
    return CCCs


class GenSVDSample(GenSample):
    """
    dataset_settings = luigi.DictParameter()
        dataset_name : str
            Name of the dataset to load
        part_fraction : float in [0,1]
            What percent of the dataset does each party have
        doc_ind : int
            The index of the doc for which to generate samples
        SVD_type : Union['hidden_eigs', 'exposed_eigs', 'full_correlation']
            type of distributed SVD to simulate
        SVD_k : int
            rank of partial SVD

    random_seed = luigi.Parameter()
    generate_positive_sample = luigi.BoolParameter()
    sample_number = luigi.IntParameter()
    """

    @property
    def X(self) -> csr_matrix:
        if not hasattr(self, 'Xtr'):
            ds = load_dataset(self.dataset_settings['dataset_name'])
            self.Xtr = ds['Xtr'].astype(np.double)
            k = self.dataset_settings['SVD_k']
            if issparse(self.Xtr):
                self.svd = partial(linalg.svds, k=k,
                                   return_singular_vectors=True)
                self.vstack = sparse_vstack
                self.format_x = lambda x: x
                self.toarray = lambda x: x.toarray()
            else:
                def full_rank_k_svd(X, k):
                    U, S, Vt = np.linalg.svd(X, full_matrices=False,
                                             compute_uv=True)
                    S = S[0:k]
                    Vt = Vt[0:k, :]
                    U = U[:, 0:k]
                    return U, S, Vt
                self.svd = partial(full_rank_k_svd, k=k)
                self.vstack = np.vstack
                self.format_x = lambda x: x[np.newaxis, :]
                self.toarray = lambda x: x
        return self.Xtr

    def gen_sample(self, sample_number: int):
        n = self.X.shape[0]
        Inds = gen_attacker_and_defender_indices(
            n,
            self.dataset_settings['part_fraction'],
            self.dataset_settings['doc_ind'],
            self.random_seed+'sample{}'.format(sample_number)
            )
        I_atk = Inds['I_attacker']
        if self.generate_positive_sample:
            I_def = Inds['I_defender_with']
        else:
            I_def = Inds['I_defender_without']

        X_def = self.X[I_def, :]
        X_atk = self.X[I_atk, :]

        Xs = self.vstack((X_atk, X_def))

        if self.dataset_settings['SVD_type'] == 'hidden_eigs':
            U, S, Vt = self.svd(Xs)
            Us, Ss, Vts = self.svd(X_atk)
            samp_ratio = Xs.shape[0] / float(X_atk.shape[0])
            S2_est = Ss ** 2 * samp_ratio  # best estimator I could find
            XTX_est = np.dot(Vt.T * S2_est, Vt)

        elif self.dataset_settings['SVD_type'] == 'exposed_eigs':
            U, S, Vt = self.svd(Xs)
            XTX_est = np.dot(Vt.T * S, Vt)

        elif self.dataset_settings['SVD_type'] == 'full_correlation':
            XTX_est = (Xs.T).dot(Xs)

        else:
            raise NotImplementedError('SVD_type={} is not implemented.'.format(
                self.dataset_settings['SVD_type']))

        x = self.format_x(self.X[self.dataset_settings['doc_ind'], :])

        wt = self.toarray(x.T.dot(x))
        # elementwise weighted statistics
        XW = np.abs(XTX_est * wt)

        # stat_names = []
        stats = []

        # stat_names.append('weight_correlation_max')
        stats.append(np.max(XW))
        # stat_names.append('weight_correlation_sum_l1')
        stats.append(np.sum(XW))
        # stat_names.append('weight_correlation_sum_l2')
        stats.append(np.sum(XW * XW))

        # normed difference statistics
        xw_fro = np.linalg.norm(XW, 'fro')
        wt_fro = np.linalg.norm(wt, 'fro')
        assert wt_fro > 0
        if xw_fro == 0.0:
            XW = np.abs(wt / wt_fro)
        else:
            XW = np.abs(XTX_est / xw_fro - wt / wt_fro)
        # stat_names.append('weighted_difference_max')
        stats.append(np.max(XW))
        # stat_names.append('weighted_difference_sum_l1')
        stats.append(np.sum(XW))
        # stat_names.append('weighted_difference_sum_l2')
        stats.append(np.sum(XW * XW))

        stats = np.array(stats)[np.newaxis, :]
        y = 1 if self.generate_positive_sample else 0
        return stats, np.array([y])


class GenSamplesSVD(
    GenSamples(GenSVDSample, x_concatenator=np.vstack, generate_in_batch=True)
    ):
    pass


class FitKNNModelSVD(
    KNNFitterMixin(neighbor_method='sqrt'),
    FitModel(GenSamplesSVD)
    ):
    pass


class EvaluateKNNSVDSD(
    EvaluateStatisticalDistance(samplegen=GenSamplesSVD, model=FitKNNModelSVD)
    ):
    pass


class CCCSVD(ComputeConvergenceCurve(EvaluateKNNSVDSD)):
    pass
