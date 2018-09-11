import pytest
import numpy as np

from dataset_utils import regression_datasets, text_datasets, recsys_datasets


@pytest.mark.parametrize('name', ['MillionSongs', 'OnlineNewsPopularity'])
def test_load_regression_dataset(name):
    ds = regression_datasets.load_dataset(name)
    assert ds['Xtr'].size > 1000


@pytest.mark.parametrize('name', ['20NG'])
def test_load_text_dataset(name):
    ds = text_datasets.load_dataset(name)
    n, d = ds['Xtr'].shape
    assert np.sum(np.abs(np.asarray(ds['Xtr'].sum(1))-1)) < 1
    print(n,d)
    assert n > 5000 and d > 1000


@pytest.mark.parametrize('name', ['ml-1m'])
def test_load_rs_dataset(name):
    ds = recsys_datasets.load_dataset(name)
    n, d = ds['Xtr'].shape
    assert n == 6040 and d == 3706 and ds['Xtr'].shape == ds['Xte'].shape


def test_get_ml1m_user():
    # 1::1193::5::978300760
    # 1::661::3::978302109
    # 1::914::3::978301968
    # 1::3408::4::978300275
    vals = recsys_datasets.get_ml1m_user(0)
    assert vals['User'].Age[0] == 1
    R = vals['Ratings']

    assert R[R.MovieID == 1193].rating.iloc[0] == 5
    assert R[R.MovieID == 661].rating.iloc[0] == 3
    assert R[R.MovieID == 914].rating.iloc[0] == 3
    assert R[R.MovieID == 3408].rating.iloc[0] == 4
    print(vals)

def test_get_twenty_doc():
    x = text_datasets.get_twenty_doc(0)
    for (score, word) in x['words']:
        assert word in x['text'].lower()