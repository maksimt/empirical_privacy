import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split
from joblib import Memory

from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR


memory = Memory(cachedir=LUIGI_COMPLETED_TARGETS_DIR, verbose=0)


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
    UI, R, n, d = get_ml1m_ratings()

    UItr, UIte, Rtr, Rte = train_test_split(
        UI,
        R,
        test_size=test_size,
        random_state=random_state,
        stratify=None
        )

    Xtr = sp.sparse.coo_matrix((Rtr, (UItr[:, 0], UItr[:, 1])),
                               shape=(n, d)).tocsr()

    Xte = sp.sparse.coo_matrix((Rte, (UIte[:, 0], UIte[:, 1])),
                               shape=(n, d)).tocsr()

    return {'Xtr': Xtr, 'Xte': Xte, 'ytr': None, 'yte': None}

def get_ml1m_user(user_ind, test_size=0.3, random_state=0):
    UI, R, n, d, I_ind = get_ml1m_ratings()

    I_ind_inv = {v: k for (k, v) in I_ind.items()}

    UItr, UIte, Rtr, Rte = train_test_split(
        UI,
        R,
        test_size=test_size,
        random_state=random_state,
        stratify=None
        )

    U_user = UItr[:, 0] == user_ind
    movies_rated = UItr[U_user, 1]
    ratings = Rtr[U_user]

    DF_movies = pd.DataFrame.from_csv(
        '/datasets/ml-1m/movies.dat',
        sep='::',
        index_col=None,
        header=None,
        )
    DF_movies.columns = ['MovieID', 'Title', 'Genres']

    DF_users = pd.DataFrame.from_csv(
        '/datasets/ml-1m/users.dat',
        sep='::',
        index_col=None,
        header=None
        )
    DF_users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']

    User_data = DF_users[DF_users.UserID == user_ind + 1]
    Movies_rated = DF_movies[
        DF_movies.MovieID.isin(I_ind_inv[j] for j in movies_rated)]
    Movies_rated.loc[:, 'rating'] = \
        [ratings[movies_rated == I_ind[i]][0] for i in Movies_rated.MovieID]

    return {'User':User_data, 'Ratings':Movies_rated}

@memory.cache
def get_ml1m_ratings():
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

    U_matrix = DF.loc[:, 'UserID'].values
    I_matrix = DF.loc[:, 'MovieID'].values
    R_matrix = DF.loc[:, 'Rating'].values
    I_matrix = list(map(lambda i: I_ind[i], I_matrix))
    U_matrix = list(map(lambda u: U_ind[u], U_matrix))

    n = R_matrix.size
    UI = np.hstack((np.array(U_matrix[0:n]).reshape(n, 1),
                    np.array(I_matrix[0:n]).reshape(n, 1)))
    R = np.array(R_matrix[0:n])

    n, d = len(np.unique(U)), len(np.unique(I))

    return UI, R, n, d, I_ind