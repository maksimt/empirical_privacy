from copy import deepcopy
from functools import partial
from itertools import product

import numpy as np
from scipy.sparse import csr_matrix, linalg, vstack as sparse_vstack, issparse
import luigi

from dataset_utils.common import load_dataset
from empirical_privacy.row_distributed_common import \
    gen_attacker_and_defender_indices
from experiment_framework.privacy_estimator_mixins import KNNFitterMixin,\
    ExpectationFitterMixin
from experiment_framework.sampling_framework import GenSample, GenSamples, FitModel, \
    EvaluateStatisticalDistance, ComputeConvergenceCurve
from experiment_framework.asymptotic_analysis import ComputeAsymptoticAccuracy
from experiment_framework.utils.helpers import AllAsymptotics


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

def svd_asymptotic_settings(n_docs=5,
                            n_max=2**12):
    setv = {
        'gen_sample_kwargs'             : {
            'generate_in_batch': True,
            'x_concatenator'   : 'numpy.vstack'
            },
        'fitter'                        : 'knn',
        'fitter_kwargs'                 : {
            'neighbor_method': 'gyorfi'
            },
        'n_docs'                        : n_docs,
        'n_trials_per_training_set_size': 30,
        'n_max'                         : n_max,
        'validation_set_size'           : 2**10
    }
    return setv

def svd_reqs(datasets=None, part_fractions=None, svd_kwargs=None):
    if datasets is None:
        datasets = ['ml-1m', '20NG']#, 'ml-1m']
    if part_fractions is None:
        part_fractions = [0.01, 0.1, 0.9]  # [0.01, 0.1]
    if svd_kwargs is None:
        svd_kwargs = dict()
    reqs = []
    aas = svd_asymptotic_settings()
    for ds, pf in product(datasets, part_fractions):
        dss = svd_dataset_settings(dataset_name=ds,
                                   part_fraction=pf,
                                   **svd_kwargs)
        reqs += AllAsymptotics(
            gen_sample_path='empirical_privacy.row_distributed_svd.GenSVDSample',
            dataset_settings = dss,
            asymptotic_settings = aas
        ).requires()
    return reqs


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

    def _gen_Xs(self, sample_number: int) -> np.ndarray:
        n = self.X.shape[0]
        Inds = gen_attacker_and_defender_indices(
            n,
            self.dataset_settings['part_fraction'],
            self.dataset_settings['doc_ind'],
            '{}sample{}'.format(self.random_seed, sample_number)
            )
        I_atk = Inds['I_attacker']
        if self.generate_positive_sample:
            I_def = Inds['I_defender_with']
        else:
            I_def = Inds['I_defender_without']

        X_def = self.X[I_def, :]
        X_atk = self.X[I_atk, :]

        return X_atk, X_def

    def _gen_XTX_est(self, sample_number: int) -> np.ndarray:
        X_atk, X_def = self._gen_Xs(sample_number)
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
        return XTX_est

    def gen_sample(self, sample_number: int):
        XTX_est = self._gen_XTX_est(sample_number)

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


class GenFullViewSVDSample(GenSVDSample):
    """
    Generate SVD samples where the sample is the entire V^T matrix reshaped
    as a vector
    """
    def gen_sample(self, sample_number: int):
        Xs = self._gen_Xs(sample_number)
        U, S, Vt = self.svd(Xs)
        y = 1 if self.generate_positive_sample else 0
        return Vt.ravel(), np.array([y])


class GenSamplesSVD(
    GenSamples(GenSVDSample, x_concatenator=np.vstack, generate_in_batch=True)
    ):
    pass

class GenFVSamplesSVD(
    GenSamples(GenFullViewSVDSample,
               x_concatenator=np.vstack,
               generate_in_batch=True)
    ):
    pass

class FitKNNModelSVD(
    KNNFitterMixin(neighbor_method='gyorfi'),
    FitModel(GenSamplesSVD)
    ):
    pass

class FitExpModelSVD(
    ExpectationFitterMixin(statistic_column=4),
    FitModel(GenSamplesSVD)
    ):
    pass

class FitKNNFVModelSVD(
    KNNFitterMixin(neighbor_method='gyorfi'),
    FitModel(GenFVSamplesSVD)
    ):
    pass

class EvaluateKNNSVDSD(
    EvaluateStatisticalDistance(samplegen=GenSamplesSVD, model=FitKNNModelSVD)
    ):
    pass

class EvaluateExpSVDSD(
    EvaluateStatisticalDistance(samplegen=GenSamplesSVD, model=FitExpModelSVD)
    ):
    pass

class EvaluateKNNFVSVDSD(
    EvaluateStatisticalDistance(samplegen=GenFVSamplesSVD,
                                model=FitKNNFVModelSVD)
    ):
    pass

class CCCSVD(ComputeConvergenceCurve(EvaluateKNNSVDSD)):
    pass


class ExpCCCSVD(ComputeConvergenceCurve(EvaluateExpSVDSD)):
    pass

class CCCFVSVD(ComputeConvergenceCurve(EvaluateKNNFVSVDSD)):
    pass

class AsymptoticAnalysisSVD(ComputeAsymptoticAccuracy(CCCSVD)):
    pass

def gen_SVD_asymptotics_for_multiple_docs(n_bootstraps=100,
                                          p=0.99,
                                          n_docs=10,
                                          n_trials_per_training_set_size=3,
                                          validation_set_size=64,
                                          n_max=256,
                                          dataset_settings=None,
                                          AAType = AsymptoticAnalysisSVD
                                          ):
    if dataset_settings is None:
        dataset_settings = svd_dataset_settings()

    AAs = []
    for doc_i in range(n_docs):
        ds = deepcopy(dataset_settings)
        ds['doc_ind'] = doc_i
        AAs.append(AAType(
            n_bootstraps=n_bootstraps,
            confidence_interval_prob=p,
            n_trials_per_training_set_size=n_trials_per_training_set_size,
            n_max=n_max,
            dataset_settings=ds,
            validation_set_size=validation_set_size
            )
            )
    return AAs


class AllSVDAsymptotics(luigi.WrapperTask):
    datasets = luigi.Parameter(default=None)
    part_fractions = luigi.Parameter(default=None)
    svd_kwargs = luigi.DictParameter(default=None)

    def requires(self):
        return svd_reqs(datasets=self.datasets,
                        part_fractions=self.part_fractions,
                        svd_kwargs=self.svd_kwargs)
