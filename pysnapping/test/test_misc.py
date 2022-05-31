import pytest

import numpy as np
from numpy.testing import assert_equal

from pysnapping.shapes import (
    GeomWithDists,
    shape_dists_trusted_at,
    simplify_rdp_2d_keep_z,
)


@pytest.mark.parametrize(
    "ls_trusted,expected",
    [("1111", "011111110"), ("0000", "0" * 9), ("0011", "000001110")],
)
def test_shape_dists_trusted_at(ls_trusted, expected):
    ls_trusted = [c == "1" for c in ls_trusted]
    expected = [c == "1" for c in expected]
    ls_dists = [-10, 1, 3, 11]
    dists = [-10.1, -10, 0, 1, 2, 3, 5, 11, 12]
    shape = GeomWithDists(None, ls_dists, ls_trusted, None, None)
    result = shape_dists_trusted_at(shape, dists)
    assert np.all(result == expected)


def test_simplify():
    coords = [[1, 2, float("nan")], [2.99, 3.99, 1000], [3, 4, 5]]
    simple_coords = simplify_rdp_2d_keep_z(coords, tolerance=0.1)
    # assert_equal to be able to compare with NaNs
    assert_equal(simple_coords, np.array(coords)[[0, 2]])
