import numpy as np


def gen_random_subset(n, n_part, seed_str, i_exclude=[], i_include=[]):
    """
    Generate a random subset of indices.
    Parameters
    ----------
    n: int
        largest possible index
    n_part: int
        how many indices to select
    seed_str: str
        seed string from for PRNG
    i_exclude: int or list of ints
        which indices must be excluded
    i_include: int or list of ints
        which indices must be included

    Returns
    -------
    rtv: list
        list of indexes
    """
    S = set(range(n))
    if not hasattr(i_exclude,'__iter__'):
        i_exclude=[i_exclude]
    if not hasattr(i_include,'__iter__'):
        i_include=[i_include]

    if i_exclude:
        S -= set(i_exclude)
    S = list(S)
    seed_val = hash(seed_str) % 4294967296  # max seed +1
    np.random.seed(hash(seed_val))
    np.random.shuffle(S)  # this modifies S
    rtv = []
    if i_include:
        n_part -= len(i_include)
        rtv = i_include
    rtv = rtv+S[0:(n_part)]
    np.random.shuffle(rtv)
    return rtv

def gen_attacker_and_defender_indices(n, part_fraction, doc_ind, seed):
    n_part = int(round(n * part_fraction))

    I_defender_without_doc = gen_random_subset(n, n_part, seed +  'without',
                                             i_exclude=doc_ind)
    I_defender_with_doc = gen_random_subset(n, n_part, seed + 'with',
                                          i_exclude=doc_ind)
    I_defender_with_doc[0] = doc_ind
    I_attacker = gen_random_subset(n, n_part, seed+'adversary')

    I_defender_with_doc = sorted(I_defender_with_doc)
    I_defender_without_doc = sorted(I_defender_without_doc)
    I_attacker = sorted(I_attacker)

    return {'I_defender_with':I_defender_with_doc,
            'I_defender_without':I_defender_without_doc,
            'I_attacker':I_attacker
    }


# class ComputeBaseModel(luigi.Target,
#        AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
#        LoadInputDictMixin
#     ):
#     dataset_name = luigi.Parameter()
#     part_percent = luigi.NumericalParameter(var_type=float, min_value=0,
#                                             max_value=100)
#     problem_settings = luigi.DictParameter()
#
#     adversary_seed = luigi.IntParameter()
#
#     doc_ind = luigi.IntParameter()  # which document are we checking privacy for
#     x_with_seed = luigi.IntParameter()  # which trial for this particular
#     x_without_seed = luigi.IntParameter()  # which trial for this particular
#
#     @property
#     def priority(self):
#         return 1.0 / self.part_percent
#
#     def run(self):
#         logger.info('Loading data {}'.format(self.dataset_name))
#         if self.problem_settings['type'] == "TM" or \
#                 self.problem_settings['dataset_name'] in ['Enron',
#                                                           'Reuters',
#                                                           '20NG']:
#             ds = datasets.load_dataset(self.dataset_name,
#                                        min_words_per_doc=min_doc_length,
#                                        dict_size=n_terms_lra)
#             X = ds['X'].toarray()
#
#         elif self.problem_settings['dataset_name'] in ['ML-1M', 'Yelp']:
#             ds = datasets.load_recsys_dataset(self.dataset_name)
#             X = sp.sparse.coo_matrix(
#                 (ds['R'], (ds['UI'][:, 0], ds['UI'][:, 1])),
#                 shape=(ds['n'], ds['d'])).toarray()
#         elif self.problem_settings['dataset_name'] in ['MillionSongs']:
#             fn = get_cached_name(datasets.load_regression_dataset,
#                                  self.dataset_name)
#             if not fn:
#                 ds = datasets.load_regression_dataset(self.dataset_name)
#                 X = ds['Xtr']
#             else:
#                 f = h5py.File(fn, 'r')
#                 X = f['Xtr']
#             # X = X - np.mean(X, 0)
#         else:
#             raise ValueError('Unrecognized dataset {}'.format(
#                 self.problem_settings['dataset_name']))
#
#         logger.info('Generating parts')
#         n, d = X.shape
#
#         n_part = int(round(n * self.part_percent))
#
#         I_victim_without_doc = gen_random_subset(n, n_part,
#                                                  str(
#                                                      self.x_without_seed) + 'without',
#                                                  i_exclude=self.doc_ind)
#         I_victim_with_doc = gen_random_subset(n, n_part,
#                                               str(
#                                                   self.x_with_seed) + 'with',
#                                               i_exclude=self.doc_ind)
#         I_victim_with_doc[0] = self.doc_ind
#         I_adversary = gen_random_subset(n, n_part,
#                                         str(self.adversary_seed))
#
#         Xs = {}
#         I_victim_with_doc = sorted(I_victim_with_doc)
#         I_victim_without_doc = sorted(I_victim_without_doc)
#         I_adversary = sorted(I_adversary)
#
#         logger.info('Loading X')
#         X = X[...]
#         x = X[self.doc_ind, :]
#         x = x.reshape((1, x.size))
#
#         logger.info('Performing selection of I_adv')
#
#         X_adversary = X[I_adversary, :]
#
#         logger.info('Performing selection of I_with')
#
#         Xs['with'] = np.vstack((X[I_victim_with_doc, :],
#                                 X_adversary
#                                 )
#                                )
#         logger.info('Performing selection of I_without')
#         Xs['without'] = np.vstack((X[I_victim_without_doc, :],
#                                    X_adversary
#                                    )
#                                   )
#         X_full = X
#         X = Xs
#         logger.info('Done selecting data')
#         soln = {}
#
#         for kv in X:
#             if self.problem_settings['type'] == "TM":
#                 k = self.problem_settings['k_value']
#                 n, d = X[kv].shape
#                 Model = NMF_TM_Estimator(n, d, k,
#                                          handle_tfidf=True,
#                                          handle_normalization=True,
#                                          nmf_kwargs={
#                                              'store_intermediate': True
#                                          },
#                                          max_iter=33
#                                          )
#                 soln[kv] = Model.fit_model(X[kv])
#                 soln[kv].sparsify()
#
#             if self.problem_settings['type'] == "SVD1":
#                 # here X.T * X is simply shared
#                 XtX = np.dot(X[kv].T, X[kv])
#                 # ew, ev = np.linalg.eig(XtX)
#                 # ew = ew[0:k]
#                 # ev = ev[:, 0:k]
#                 soln[kv] = XtX
#
#             if self.problem_settings['type'] == "SVD2":
#                 # like SVD4 but we get the full spectrum
#                 U, S, Vt = np.linalg.svd(X[kv], full_matrices=0)
#                 # I = gen_random_subset(n, X[kv].shape[0], 'sample')
#                 Xs = X_full[I_adversary, :]
#                 Us, Ss, Vts = np.linalg.svd(Xs, full_matrices=0)
#                 samp_ratio = X[kv].shape[0] / float(Xs.shape[0])
#                 S2_est = Ss ** 2 * samp_ratio  # best estimator I could find
#                 k_min = min(S2_est.size, Vt.shape[0])
#                 S2_est = S2_est[0:k_min]
#                 Vt = Vt[0:k_min, :]
#                 XtX_est = np.dot(Vt.T * S2_est, Vt)
#                 # np.dot(np.dot(Vt.T, Se), Vt)
#                 soln[kv] = XtX_est
#
#             if self.problem_settings['type'] == 'SVD3':
#                 # here the eigenvalues are not hidden
#                 U, S, Vt = np.linalg.svd(X[kv], full_matrices=0)
#                 k = self.problem_settings['k']
#                 S = S[0:k]
#                 Vt = Vt[0:k, :]
#                 XTX_est = np.dot(Vt.T * S, Vt)
#                 soln[kv] = XTX_est
#
#             if self.problem_settings['type'] == 'DP_SVD':
#                 # here the eigenvalues are not hidden
#                 Mod = BlockIterSVD(k=self.problem_settings['k'],
#                                    eps_diff_priv=self.problem_settings[
#                                        'eps'])
#                 Mod = Mod.fit_model(X[kv])
#                 S = Mod.s_
#                 Vt = Mod.V_.T
#
#                 try:
#                     XTX_est = np.dot(Vt.T * S, Vt)
#                 except Exception:
#                     import pdb;
#                     pdb.set_trace()
#                 soln[kv] = XTX_est
#
#             if self.problem_settings['type'] in ['SVD4', 'lra', 'pcr']:
#                 # this simulates hiding the eigenvalues in the distributed
#                 # computation; the adversary will try to estimate them using
#                 # their own data or bootstrap
#                 # difference between this and SVD4 is that here we only use
#                 # the top-k singular values
#                 logger.info('Computing SVD of X[kv]')
#                 U, S, Vt = np.linalg.svd(X[kv], full_matrices=0)
#                 k = self.problem_settings['k']
#                 Xs = X_full[I_adversary, :]
#                 logger.info('Computing SVD of Xs')
#                 Us, Ss, Vts = np.linalg.svd(Xs, full_matrices=0)
#                 samp_ratio = X[kv].shape[0] / float(Xs.shape[0])
#                 S2_est = Ss ** 2 * samp_ratio  # best estimator I could find
#                 S2_est = S2_est[0:k]
#                 Vt = Vt[0:k, :]
#                 logger.info('Computing X^T X')
#                 XtX_est = np.dot(Vt.T * S2_est, Vt)
#                 soln[kv] = XtX_est
#
#             if self.problem_settings['type'] == "RS":
#                 k = self.problem_settings['k_value']
#                 n, d = X[kv].shape
#                 Model = NMF_RS_Estimator(n, d, k, max_iter=33,
#                                          nmf_kwargs={
#                                              'store_intermediate': True
#                                          })
#                 soln[kv] = Model.fit_from_Xtr(X[kv])
#                 # soln[kv].sparsify()
#
#         rtv = {
#             'doc_ind': self.doc_ind,
#             'x': x,
#             'x_with_seed': self.x_with_seed,
#             'x_without_seed': self.x_without_seed,
#             'I_victim_with_doc': I_victim_with_doc,
#             'I_victim_without_doc': I_victim_without_doc,
#             'I_adversary': I_adversary,
#             'soln': soln
#         }
#
#         # drop all but the last iterations
#         if self.problem_settings['type'] == "RS" or \
#                 self.problem_settings['type'] == "TM":
#             for kv in rtv['soln']:
#                 key_d = rtv['soln'][kv].nmf_outputs['denom_W'].keys()
#                 key_n = rtv['soln'][kv].nmf_outputs['numer_W'].keys()
#                 max_d = np.max(key_d)
#                 max_n = np.max(key_n)
#                 rtv['soln'][kv].nmf_outputs['denom_W'] = rtv['soln'][
#                     kv].nmf_outputs['denom_W'][max_d]
#                 rtv['soln'][kv].nmf_outputs['numer_W'] = rtv['soln'][
#                     kv].nmf_outputs['numer_W'][max_n]
#
#         with self.output().open('w') as f:
#             datasets.dump(rtv, f, 2)  # use protocol 2 for efficiency