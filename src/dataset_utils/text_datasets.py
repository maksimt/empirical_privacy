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
        return {
            'Xtr': Xtr, 'ytr': twenty_tr.target,
            'Xte': Xte, 'yte': twenty_te.target
        }
