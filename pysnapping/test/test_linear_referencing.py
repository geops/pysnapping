import numpy as np
import pytest

from pysnapping.linear_referencing import locate, Location


def test_locate():
    assert locate(np.array([1, 2, 4]), 2.5) == Location(1, pytest.approx(0.25))
    assert locate(np.array([1, 2, 4]), 1) == Location(0, 0.0)
    assert locate(np.array([1, 2, 4]), 4) == Location(1, 1.0)
    assert locate(np.array([1, 2, 4]), 2) == Location(1, 0.0)
    assert locate(np.array([1, 2, 4]), 0.99) == Location(0, 0.0)
    assert locate(np.array([1, 2, 4]), 4.01) == Location(1, 1.0)
    assert locate(np.array([1, 1, 2, 4]), 1) == Location(0, 0.5)
    assert locate(np.array([1, 2, 4, 4]), 4) == Location(2, 0.5)
    assert locate(np.array([1, 2, 4, 4, 5]), 4) == Location(2, 0.5)
    assert locate(np.array([1, 2, 4, 4, 4, 5]), 4) == Location(3, 0.0)
    assert locate(np.array([1, 2, 4, 4, 4]), 4) == Location(3, 0.0)
    assert locate(np.array([1, 1, 1, 2, 4]), 1) == Location(1, 0.0)
    assert locate(np.array([0, 1, 1, 1, 2, 4]), 1) == Location(2, 0.0)
    assert locate(np.array([0, 1, 1, 1, 1, 2, 4]), 1) == Location(2, 0.5)
    assert locate(np.array([1, 1, 1, 1, 2, 4]), 1) == Location(1, 0.5)
    assert locate(np.array([1, 2, 4, 4, 4, 4]), 4) == Location(3, 0.5)
    assert locate(np.array([1, 2, 4, 4, 4, 4, 5]), 4) == Location(3, 0.5)
    assert locate(np.array([1, 1, 2, 4]), 0.99) == Location(0, 0.0)
    assert locate(np.array([1, 2, 4, 4]), 4.01) == Location(2, 1.0)
    assert locate(np.array([1, 1, 1, 2, 4]), 0.99) == Location(0, 0.0)
    assert locate(np.array([1, 2, 4, 4, 4]), 4.01) == Location(3, 1.0)
