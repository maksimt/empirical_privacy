import pytest
from empirical_privacy import one_bit_sum

def test_pytest():
    assert 3 == (2+1)

def test_import():
    X0, X1, y0, y1 = one_bit_sum.gen_data(10, 0.5, 100, 0)
    assert X0.shape[0] == 100