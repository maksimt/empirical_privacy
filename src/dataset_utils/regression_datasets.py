import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_dataset(dataset_name):
    available_names = ['MillionSongs', 'OnlineNewsPopularity', 'PCR_Test']
    if dataset_name is None:
        print('Warning no dataset_name provided, loading MillionSongs')
        dataset_name = 'MillionSongs'
    if dataset_name not in available_names:
        raise ValueError('Error dataset "{0}" not available, only {1} are '
                         'available'. \
            format(
            dataset_name,
            available_names
            ))

    if dataset_name == 'MillionSongs':
        DF = pd.DataFrame.from_csv('/datasets/YearPredictionMSD.txt')
        X = DF.as_matrix()
        y = X[:, 0]
        X = X[:, 1:]
        ntrain = 463715  # first ntrain rows are defined as training on UCI site

        Xtr = X[0:ntrain, :]
        ytr = y[0:ntrain]
        Xte = X[ntrain:, :]
        yte = y[ntrain:]

    elif dataset_name == 'OnlineNewsPopularity':
        DF = pd.DataFrame.from_csv(
            '/datasets/OnlineNewsPopularity/OnlineNewsPopularity.csv')
        DF_X = pd.DataFrame.sort_index(DF.drop(labels=' shares', axis=1))
        DF_y = pd.Series.sort_index(DF.loc[:, ' shares'])
        assert np.all(DF_X.index == DF_y.index)
        X = DF_X.as_matrix()
        y = DF_y.as_matrix()
        ntrain = 30000
        Xtr = X[0:ntrain, :]
        ytr = y[0:ntrain]
        Xte = X[ntrain:, :]
        yte = y[ntrain:]

    elif dataset_name == 'PCR_Test':
        np.random.seed(0)
        X = np.dot(np.random.rand(120, 11), np.random.rand(11, 50))
        X = X - np.mean(X, 0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt.T
        V = V[:, 0:10]
        b = np.arange(10)
        y = np.dot(np.dot(X, V), b) + np.random.randn(120) * 0.01
        Xte, yte = X[100:, :], y[100:]
        Xtr, ytr = X[0:100, :], y[0:100]

    return {
        'Xtr': Xtr - np.mean(Xtr, 0),
        'ytr': ytr,
        'Xte': Xte - np.mean(Xte, 0),
        'yte': yte
        }
