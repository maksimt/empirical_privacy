import numpy as np
from scipy.sparse import csr_matrix


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


def load_dataset(dataset_name):
    available_names = ['20NG']
    if dataset_name is None:
        print('Warning no dataset_name provided, loading MillionSongs')
        dataset_name = '20NG'
    if dataset_name not in available_names:
        raise ValueError(
            'Error dataset "{0}" not available, only {1} are available'. \
                format(
                dataset_name,
                available_names
                ))

    if dataset_name == '20NG':
        twenty_tr = fetch_20newsgroups(shuffle=True, subset='train',
                                       random_state=1, data_home='/datasets',
                                       remove=('headers', 'footers', 'quotes'))
        twenty_te = fetch_20newsgroups(shuffle=True, subset='test',
                                       random_state=1, data_home='/datasets',
                                       remove=('headers', 'footers', 'quotes'))

        tf_vectorizer = CountVectorizer(max_df=0.1, min_df=10,
                                        stop_words='english')

        Xtr = tf_vectorizer.fit_transform(twenty_tr.data)
        Xte = tf_vectorizer.fit_transform(twenty_te.data)

        Xtr, I_rows_tr, I_cols_tr = _remove_zero_rows_cols(Xtr, min_row=100,
                                                           min_col=100)
        Xte, I_rows_te, I_cols_te = _remove_zero_rows_cols(Xte, min_row=100,
                                                           min_col=0)
        Xte = Xte[:, I_cols_tr]

        Xtr = csr_matrix(_normalize(Xtr))
        Xte = csr_matrix(_normalize(Xte))

        return {
            'Xtr': Xtr, 'ytr': twenty_tr.target[I_rows_tr],
            'Xte': Xte, 'yte': twenty_te.target[I_rows_te]
        }

def _normalize(X, axis=1):
    return X / (X.sum(axis) + np.spacing(10))

def _remove_zero_rows_cols(X, min_row=1, min_col=1):
    """Remove rows and columns of X that sum to 0

    Parameters
    ----------
    X : arraylike
    users * items matrix

    Returns
    -------
    X : arraylike
    user * items matirx with zero rows and columns removed

    I_users : arraylike
    indices of non-zero users

    I_items : arraylike
    indices of non-zero items

    """
    M = X>0
    I_users = np.argwhere(M.sum(1) >= min_row).ravel()
    I_items = np.argwhere(M.sum(0) >= min_col).ravel()
    X = X[I_users, :]

    X = X[:, I_items]

    return X, I_users, I_items