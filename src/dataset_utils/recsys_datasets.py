import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split


def load_dataset(dataset_name):
    available_names = ['ml-1m']
    if dataset_name is None:
        print('Warning no dataset_name provided, loading MillionSongs')
        dataset_name = 'ml-1m'
    if dataset_name not in available_names:
        raise ValueError('Error dataset "{0}" not available, only {1} are '
                         'available'. \
            format(
            dataset_name,
            available_names
            ))

    if dataset_name == 'ml-1m':
        return load_ml1m()


def load_ml1m(test_size=0.3, random_state=0):
    file_path = '/datasets/ml-1m/ratings.dat'

    DF = pd.DataFrame.from_csv(file_path, sep='::', index_col=None, header=None)
    DF.columns = pd.Index(['UserID', 'MovieID', 'Rating', 'Timestamp'],
                          dtype='int64')
    U = DF.loc[:, 'UserID'].unique()
    I = DF.loc[:, 'MovieID'].unique()

    U_arg = np.sort(U)
    I_arg = np.sort(I)
    U_ind = {}
    I_ind = {}
    for (pos, u) in enumerate(U_arg):
        U_ind[u] = pos
    for (pos, i) in enumerate(I_arg):
        I_ind[i] = pos
    U_matrix = DF.loc[:, 'UserID'].as_matrix()
    I_matrix = DF.loc[:, 'MovieID'].as_matrix()
    R_matrix = DF.loc[:, 'Rating'].as_matrix()
    I_matrix = list(map(lambda i: I_ind[i], I_matrix))
    U_matrix = list(map(lambda u: U_ind[u], U_matrix))

    n = R_matrix.size
    UI = np.hstack((np.array(U_matrix[0:n]).reshape(n, 1),
                    np.array(I_matrix[0:n]).reshape(n, 1)))
    R = np.array(R_matrix[0:n])

    UItr, UIte, Rtr, Rte = train_test_split(
        UI,
        R,
        test_size=test_size,
        random_state=random_state,
        stratify=None
        )

    n, d = len(np.unique(U)), len(np.unique(I))

    Xtr = sp.sparse.coo_matrix((Rtr, (UItr[:, 0], UItr[:, 1])),
                               shape=(n, d)).tocsr()

    Xte = sp.sparse.coo_matrix((Rte, (UIte[:, 0], UIte[:, 1])),
                               shape=(n, d)).tocsr()

    return {'Xtr': Xtr, 'Xte': Xte, 'ytr': None, 'yte': None}
