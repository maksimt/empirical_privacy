import numpy as np
from scipy.sparse import csr_matrix, linalg

from dataset_utils.common import load_dataset
from empirical_privacy.row_distributed_common import \
    gen_attacker_and_defender_indices
from luigi_utils.privacy_estimator_mixins import KNNFitterMixin
from luigi_utils.sampling_framework import GenSample, GenSamples, FitModel, \
    EvaluateStatisticalDistance, ComputeConvergenceCurve


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
        return self.Xtr

    def gen_sample(self, sample_number: int):
        n = self.X.shape[0]

        Inds = gen_attacker_and_defender_indices(
            n,
            self.dataset_settings['part_fraction'],
            self.dataset_settings['doc_ind'],
            self.random_seed
            )
        I_atk = Inds['I_attacker']
        if self.generate_positive_sample:
            I_def = Inds['I_defender_with']
        else:
            I_def = Inds['I_defender_without']

        X_def = self.X[I_def, :]
        X_atk = self.X[I_atk, :]
        Xs = np.vstack((X_atk, X_def))

        k = self.dataset_settings['SVD_k']

        if self.dataset_settings['SVD_type'] == 'hidden_eigs':
            U, S, Vt = linalg.svds(Xs, k=k, return_singular_vectors=True)
            Us, Ss, Vts = linalg.svds(X_atk, k=k, return_singular_vectors=True)
            samp_ratio = Xs.shape[0] / float(X_atk.shape[0])
            S2_est = Ss ** 2 * samp_ratio  # best estimator I could find
            XTX_est = np.dot(Vt.T * S2_est, Vt)

        elif self.dataset_settings['SVD_type'] == 'exposed_eigs':
            U, S, Vt = linalg.svds(Xs, k=k, return_singular_vectors=True)
            XTX_est = np.dot(Vt.T * S, Vt)

        elif self.dataset_settings['SVD_type'] == 'full_correlation':
            XTX_est = (Xs.T).dot(Xs)

        else:
            raise NotImplementedError('SVD_type={} is not implemented.'.format(
                self.dataset_settings['SVD_type']))

        x = self.X[self.dataset_settings['doc_ind'], :][:, np.newaxis]
        wt = x.T.dot(x)
        # elementwise weighted statistics
        XW = np.abs(XTX_est*wt)

        # stat_names = []
        stats = []

        # stat_names.append('weight_correlation_max')
        stats.append(np.max(XW))
        # stat_names.append('weight_correlation_sum_l1')
        stats.append(np.sum(XW))
        # stat_names.append('weight_correlation_sum_l2')
        stats.append(np.sum(XW*XW))

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
        stats.append(np.sum(XW*XW))

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
