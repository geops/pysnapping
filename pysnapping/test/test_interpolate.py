from numpy.testing import assert_allclose
import numpy as np
import pytest


from pysnapping.interpolate import interp1d_assume_sorted_allow_repeated as interp


def test_interpolate():
    f = interp([0, 1, 1, 2], [[10, 1], [20, 2], [30, 3], [40, 4]], axis=0)
    assert_allclose(f(1), [25, 2.5])

    f = interp([0, 0], [1, 3])
    assert_allclose(f(0), 2)
    assert_allclose(f([0, 0]), [2, 2])
    with pytest.raises(ValueError):
        f(0.1)
    with pytest.raises(ValueError):
        f(-0.1)

    f = interp([0, 0], [1, 3], bounds_error=False)
    assert np.isnan(f(0.1))

    f = interp([0, 0, 0], [2, 3, 4], bounds_error=False, fill_value="extrapolate")
    assert_allclose(f(0), 3)
    with pytest.raises(ValueError):
        f(0.1)
    with pytest.raises(ValueError):
        f(-0.1)

    f = interp([1, 1], [1, 1], fill_value=10, bounds_error=False)
    assert f(0.99) == 10
    assert f(1.01) == 10
    assert_allclose(f([0.99, 1.01]), [10, 10])

    f = interp([1, 1], [1, 1], fill_value=(9, 10), bounds_error=False)
    assert f(0.99) == 9
    assert f(1.01) == 10
    assert_allclose(f([1.01, 1, 0.99, np.nan]), [10, 1, 9, np.nan])
