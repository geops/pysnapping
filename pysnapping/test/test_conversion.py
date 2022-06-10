import numpy as np
from numpy.testing import assert_equal

from pysnapping.conversion import simplify_2d_keep_z


def test_simplify():
    coords = [[1, 2, float("nan")], [2.99, 3.99, 1000], [3, 4, 5]]
    simple_coords = simplify_2d_keep_z(coords, tolerance=0.1)
    # assert_equal to be able to compare with NaNs
    assert_equal(simple_coords, np.array(coords)[[0, 2]])
