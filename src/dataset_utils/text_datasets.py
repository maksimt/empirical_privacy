import numpy as np
from scipy.sparse import csr_matrix
from joblib import Memory

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR


memory = Memory(cachedir=LUIGI_COMPLETED_TARGETS_DIR, verbose=0)


def load_dataset(dataset_name):
    available_names = ['20NG']
    if dataset_name is None:
        print('Warning no dataset_name provided, loading 20NG')
        dataset_name = '20NG'
    if dataset_name not in available_names:
        raise ValueError(
            'Error dataset "{0}" not available, only {1} are available'. \
                format(
                dataset_name,
                available_names
                ))

    if dataset_name == '20NG':
        return twenty_ds()

def _vectorizer() -> CountVectorizer:
    return CountVectorizer(max_df=0.1, min_df=100,
                                    stop_words='english')


@memory.cache
def twenty_ds():
    twenty_tr = fetch_20newsgroups(shuffle=True, subset='train',
                                   random_state=1, data_home='/datasets',
                                   remove=('headers', 'footers', 'quotes'))
    twenty_te = fetch_20newsgroups(shuffle=True, subset='test',
                                   random_state=1, data_home='/datasets',
                                   remove=('headers', 'footers', 'quotes'))

    tf_vectorizer = _vectorizer()
    tf_vectorizer.fit(twenty_tr.data)
    Xtr = tf_vectorizer.transform(twenty_tr.data).toarray()
    Xte = tf_vectorizer.transform(twenty_te.data).toarray()

    Xtr, I_rows_tr, I_cols_tr = _remove_zero_rows_cols(Xtr, min_row=10,
                                                       min_col=100)
    Xte, I_rows_te, I_cols_te = _remove_zero_rows_cols(Xte, min_row=10,
                                                       min_col=0)
    Xte = Xte[:, I_cols_tr]

    Xtr = csr_matrix(_normalize(Xtr))
    Xte = csr_matrix(_normalize(Xte))

    return {
        'Xtr': Xtr, 'ytr': twenty_tr.target[I_rows_tr],
        'Xte': Xte, 'yte': twenty_te.target[I_rows_te]
    }


def get_twenty_doc(doc_ind, subset='train'):
    twenty = fetch_20newsgroups(shuffle=True, subset=subset,
                                   random_state=1, data_home='/datasets',
                                   remove=('headers', 'footers', 'quotes'))
    tf_vectorizer = _vectorizer()
    tf_vectorizer.fit(twenty.data)
    X =tf_vectorizer.transform(twenty.data).toarray()
    X, I_rows_tr, I_cols_tr = _remove_zero_rows_cols(X, min_row=10,
                                                       min_col=100)
    X = _normalize(X)

    n, d = X.shape

    # idf weighting is only used to display top words in this document
    idf = np.log(n/(np.sum(X>0,0)+1))
    x_tfidf = X[doc_ind, :] * idf

    J = np.argwhere(x_tfidf>0).ravel()
    words = {ind:word for (word,ind) in tf_vectorizer.vocabulary_.items()}
    vocab = [words[j] for j in J]
    I = np.argsort(x_tfidf[J])[::-1]
    vocab = [vocab[i] for i in I]

    rtv = {'text': twenty.data[I_rows_tr[doc_ind]],
            'tfidf': x_tfidf.sum(),
            'words': list(zip(x_tfidf[J][I], vocab))}
    return rtv


def _normalize(X, axis=1):
    return X / (X.sum(axis)[:, np.newaxis] + np.spacing(10))


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