from math import ceil, sqrt, floor

import numpy as np
from scipy.integrate import quad
from scipy.stats import gaussian_kde
from sklearn import neighbors
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


def ExpectationFitterMixin(statistic_column=0, bandwidth_method=None):
    class T(DensityEstFitterMixin(statistic_column, bandwidth_method)):
        # picks up the fit_model method

        @classmethod
        def compute_classification_accuracy(self, model=None, *samples):
            assert model is not None, 'Model must be fitted first'
            f0, f1 = model['f0'], model['f1']
            X, y = _stack_samples(samples)
            X = X[:, self.statistic_column]  # we pick one statistic given a
            # multidimensional sample
            X = X.ravel()
            f0x = f0(X)
            f1x = f1(X)
            denom = (f0x + f1x + np.spacing(1))
            numer = np.abs(f0x - f1x)
            return 0.5 + 0.5 * np.mean(numer / denom)

    return T

def _assert_at_most_1dim(*Xs):
    for X in Xs:
        assert X.ndim <= 2, 'Xs must be at most 1 dimensional'
        if X.ndim == 2:
            assert X.shape[1]==1, 'Xs must be at most 1 dimensional'


def DensityEstFitterMixin(statistic_column=0, bandwidth_method=None):
    class T(object):
        def fit_model(self, negative_samples, positive_samples):
            X0 = negative_samples['X']
            X1 = positive_samples['X']
            y0 = negative_samples['y']
            y1 = positive_samples['y']

            if X0.ndim >= 2:
                X0 = X0[:, self.statistic_column]  # we pick one dimension of a
            # multi-dimensional sample
            if X1.ndim >= 2:
                X1 = X1[:, self.statistic_column]

            _assert_at_most_1dim(X0, X1)

            X0, X1 = X0.ravel(), X1.ravel()

            num_samples = y0.size + y1.size

            bw = None
            if hasattr(self.bandwidth_method, '__call__'):
                bw = float(
                    self.bandwidth_method(num_samples)) / num_samples  # eg log
            if hasattr(self.bandwidth_method, '__add__'):  # float:
                bw = num_samples ** (1 - self.bandwidth_method)
            f0 = gaussian_kde(X0, bw_method=bw)
            f1 = gaussian_kde(X1, bw_method=bw)

            return {'f0': f0, 'f1': f1}

        @classmethod
        def compute_classification_accuracy(cls, model=None, *samples):
            assert model is not None, 'Model must be fitted first'
            f0, f1 = model['f0'], model['f1']
            return 0.5 + \
                   0.5 * 0.5 * \
                   quad(lambda x: np.abs(f0(x) - f1(x)), -np.inf, np.inf)[0]

    T.bandwidth_method = bandwidth_method
    T.statistic_column = statistic_column

    # TODO: Unclear how to handle validation set here

    return T

def get_k(method, num_samples, d=None, **kwargs):
    if method == 'gyorfi':
        k = num_samples ** (2 / (d + 2))
    elif 'sqrt' in method:
        k = floor(sqrt(num_samples))

    k = int(k)
    if k % 2 == 0:  # ensure k is odd
        k += 1
    return k


def KNNFitterMixin(neighbor_method='sqrt_random_tiebreak'):
    class T(object):
        def fit_model(self, negative_samples, positive_samples):
            X0 = negative_samples['X']
            X1 = positive_samples['X']
            y0 = negative_samples['y']
            y1 = positive_samples['y']

            X0, X1 = _ensure_2dim(X0, X1)

            X = np.vstack((X0, X1))
            y = np.concatenate((y0, y1))
            num_samples = X.shape[0]
            neighbor_method = self.neighbor_method
            KNN = neighbors.KNeighborsClassifier(algorithm='brute', metric='l2')
            if hasattr(neighbor_method, 'lower'):  # string
                if neighbor_method == 'sqrt':
                    k = get_k(neighbor_method, num_samples)
                    KNN.n_neighbors = k
                if neighbor_method == 'gyorfi':
                    d = X0.shape[1]
                    self.d = d
                    k = get_k('gyorfi', num_samples, d=self.d)
                    KNN.n_neighbors = k
                if neighbor_method == 'cv':
                    param_grid = \
                        [{
                            'n_neighbors': (num_samples **
                                            np.linspace(0.1, 1, 9)).astype(
                                np.int)
                            }]
                    gs = GridSearchCV(KNN, param_grid,
                                      scoring=make_scorer(accuracy_score),
                                      cv=min([3, num_samples]))
                    gs.fit(X, y)
                    KNN = gs.best_estimator_
                if neighbor_method == 'sqrt_random_tiebreak':
                    k = get_k('sqrt', num_samples)
                    KNN.n_neighbors = k
                    X = X + np.random.rand(X.shape[0], X.shape[1]) * 0.1

            KNN.fit(X, y)
            return {'KNN': KNN}

        @classmethod
        def compute_classification_accuracy(cls, model=None, *samples):
            assert model is not None, 'Model must be fitted first'
            X, y = _stack_samples(samples)
            return model['KNN'].score(X, y)

    T.neighbor_method = neighbor_method
    return T


def _ensure_2dim(X0, X1):
    # sklearn wants X to have dimension>=2
    if X0.ndim == 1:
        X0 = X0[:, np.newaxis]
    if X1.ndim == 1:
        X1 = X1[:, np.newaxis]
    return X0, X1


def _stack_samples(samples):
    X, y = None, None
    for sample in samples:
        Xs, ys = sample['X'], sample['y']
        if X is None:
            X = Xs
        else:
            X, Xs = _ensure_2dim(X, Xs)
            X = np.vstack((X, Xs))
        if y is None:
            y = ys
        else:
            y = np.concatenate((y, ys))
    return X, y
