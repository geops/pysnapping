import pyproj
import numpy as np
from numpy.testing import assert_equal, assert_allclose

from pysnapping.util import (
    simplify_2d_keep_z,
    get_trafo,
    transform_coords,
    fix_repeated_x,
)


def test_simplify():
    coords = [[1, 2, float("nan")], [2.99, 3.99, 1000], [3, 4, 5]]
    simple_coords = simplify_2d_keep_z(coords, tolerance=0.1)
    # assert_equal to be able to compare with NaNs
    assert_equal(simple_coords, np.array(coords)[[0, 2]])


def test_transform_coords_gis_order():
    coords = [[828000, 5932500], [828001, 5932502]]
    trafo = get_trafo(pyproj.CRS.from_epsg(3857), always_xy=True)
    assert_allclose(
        transform_coords(coords, trafo), [[7.438051, 46.941312], [7.438060, 46.941325]]
    )


def test_transform_coords_strict_order():
    coords = [[828000, 5932500], [828001, 5932502]]
    trafo = get_trafo(pyproj.CRS.from_epsg(3857), always_xy=False)
    assert_allclose(
        transform_coords(coords, trafo), [[46.941312, 7.438051], [46.941325, 7.438060]]
    )


def test_fix_repeated_x():
    x = np.array([1, 2, 2, 3, 4, 4, 4], dtype=float)
    y = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]], dtype=float
    )
    fixed_x, fixed_y = fix_repeated_x(x, y)
    assert_allclose(fixed_x, [1, 2, 3, 4])
    assert_allclose(fixed_y, [[1, 2], [4, 5], [7, 8], [11, 12]])
