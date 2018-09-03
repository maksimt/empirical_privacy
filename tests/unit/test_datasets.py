import pytest
from dataset_utils import regression_datasets, text_datasets, recsys_datasets

@pytest.mark.parametrize('name', ['MillionSongs', 'OnlineNewsPopularity'])
def test_load_regression_dataset(name):
    ds = regression_datasets.load_dataset(name)
    assert ds['Xtr'].size > 1000

@pytest.mark.parametrize('name', ['20NG'])
def test_load_text_dataset(name):
    ds = text_datasets.load_dataset(name)
    n, d = ds['Xtr'].shape
    assert n == 11314 and d > 5000

@pytest.mark.parametrize('name', ['ml-1m'])
def test_load_text_dataset(name):
    ds = recsys_datasets.load_dataset(name)
    n, d = ds['Xtr'].shape
    assert n==6040 and d ==3706 and ds['Xtr'].shape == ds['Xte'].shape