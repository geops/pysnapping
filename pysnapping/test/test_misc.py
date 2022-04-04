import pytest

import numpy as np
from numpy.testing import assert_equal, assert_allclose
from shapely.geometry import LineString

from pysnapping.shapes import (
    GeomWithDists,
    shape_dists_trusted_at,
    simplify_rdp_2d_keep_z,
    make_shape,
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


@pytest.mark.parametrize(
    "lon_lat_dists,expected",
    [
        (
            [
                (-1, 0, None),
                (0, 0, 0),
                (0, 1, None),
                (1, 1, 4),
                (1, 2, None),
            ],
            GeomWithDists(
                geom=LineString([(-1, 0), (0, 0), (0, 1), (1, 1), (1, 2)]),
                dists=np.array([-2, 0, 2, 4, 6], dtype=float),
                original_dists_mask=np.array([0, 1, 0, 1, 0], dtype=bool),
                finite_dists_mask=np.ones((5,), dtype=bool),
                n_finite=5,
            ),
        ),
        (
            [
                (0, 0, 0),
                (1, 1, None),
            ],
            GeomWithDists(
                geom=LineString([(0, 0), (1, 1)]),
                dists=np.array([0, None], dtype=float),
                original_dists_mask=np.array([1, 0], dtype=bool),
                finite_dists_mask=np.array([1, 0], dtype=bool),
                n_finite=1,
            ),
        ),
        (
            [
                (0, 0, None),
                (0.5, 0.5, None),
                (1, 1, None),
            ],
            GeomWithDists(
                geom=LineString([(0, 0), (1, 1)]),
                dists=np.array([None, None], dtype=float),
                original_dists_mask=np.zeros((2,), dtype=bool),
                finite_dists_mask=np.zeros((2,), dtype=bool),
                n_finite=0,
            ),
        ),
    ],
)
def test_make_shape(lon_lat_dists, expected):
    shape = make_shape(lon_lat_dists)

    assert_allclose(shape.geom.coords, expected.geom.coords)

    # geodesic distances are used to fill dist gaps, but expected dists are a raw estimate
    # based on cartesian dists, so we have to use a high tolerance here (1 degree is really long,
    # so cartesian and geodesic distances differ quite a bit).
    assert_allclose(shape.dists, expected.dists, atol=1e-2, equal_nan=True)

    assert (shape.original_dists_mask == expected.original_dists_mask).all()
    assert (shape.finite_dists_mask == expected.finite_dists_mask).all()
    assert shape.n_finite == expected.n_finite
