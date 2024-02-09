import numpy as np
from numpy.testing import assert_equal

from pysnapping.util import simplify_2d_keep_rest, cumulative_min_and_argmin


def test_simplify() -> None:
    coords = [
        [1, 2, float("nan"), np.inf],
        [2.99, 3.99, 1000, np.nan],
        [3, 4, 5, -10000.1],
    ]
    simple_coords = simplify_2d_keep_rest(coords, tolerance=0.1)
    assert_equal(simple_coords, np.array(coords)[[0, 2]])


def test_cumulative_min_and_argmin() -> None:
    cum_min, cum_argmin = cumulative_min_and_argmin([10, 3, 4, 5, 2, 10, 0, 0])
    assert cum_min.tolist() == [10, 3, 3, 3, 2, 2, 0, 0]
    assert cum_argmin.tolist() == [0, 1, 1, 1, 4, 4, 6, 6]
