from numpy.testing import assert_allclose
import numpy as np

from pysnapping.interpolate import fix_repeated_x


def test_fix_repeated_x():
    x = np.array([1, 2, 2, 3, 4, 4, 4], dtype=float)
    y = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]], dtype=float
    )
    fixed_x, fixed_y = fix_repeated_x(x, y)
    assert_allclose(fixed_x, [1, 2, 3, 4])
    assert_allclose(fixed_y, [[1, 2], [4, 5], [7, 8], [11, 12]])
