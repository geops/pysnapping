import numpy as np
from numpy.testing import assert_equal

from pysnapping.util import simplify_2d_keep_rest


def test_simplify():
    coords = [
        [1, 2, float("nan"), np.inf],
        [2.99, 3.99, 1000, np.nan],
        [3, 4, 5, -10000.1],
    ]
    simple_coords = simplify_2d_keep_rest(coords, tolerance=0.1)
    assert_equal(simple_coords, np.array(coords)[[0, 2]])
