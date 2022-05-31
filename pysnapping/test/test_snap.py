from functools import partial

import pytest
from shapely.geometry import LineString, MultiPoint
import numpy as np
from numpy.testing import assert_allclose

from pysnapping.shapes import (
    split_shape,
    split_ls_coords,
    GeomWithDists,
    get_geodesic_line_string_dists,
)
from pysnapping.linear_referencing import snap_points_in_order, SnappingError


snap = partial(
    snap_points_in_order,
    min_dist=25.0,  # meters
    start_occupied=False,
    end_occupied=False,
    try_reverse=False,
    convergence_accuracy=1.0,  # meters
)


@pytest.fixture
def simple_ls():
    return LineString([[0, 0], [1e-3, 0], [1e-3, 2e-3], [5e-3, 2e-3]])


@pytest.fixture
def straight_ls():
    # roughly 1 km long
    return LineString([[0, 0], [1e-2, 0]])


@pytest.fixture
def straight_ls_dists():
    # not the real geodesic dists, but simplified for testing
    return np.array([0, 1000], dtype=float)


def compare_split_result(result, expected):
    assert len(result) == len(expected), "result and expected result have same length"
    for i, (r, e) in enumerate(zip(result, expected)):
        try:
            assert_allclose(r, e, verbose=True)
        except AssertionError as error:
            raise AssertionError(f"entries at index {i} almost equal") from error


def test_split_ls_coords_trivial():
    ls_coords = np.array([[1, 2, 0], [3, 4, 1]])
    result = split_ls_coords(ls_coords, ls_coords[:, 2], [])
    expected = [ls_coords]
    compare_split_result(result, expected)


def test_split_ls2_coords_start():
    ls_coords = np.array([[1, 2, 10], [3, 4, 11]])
    result = split_ls_coords(ls_coords, ls_coords[:, 2], [10])
    expected = [[[1, 2, 10], [1, 2, 10]], [[1, 2, 10], [3, 4, 11]]]
    compare_split_result(result, expected)


def test_split_ls2_coords_end():
    ls_coords = np.array([[1, 2, 10], [3, 4, 11]])
    result = split_ls_coords(ls_coords, ls_coords[:, 2], [11])
    expected = [[[1, 2, 10], [3, 4, 11]], [[3, 4, 11], [3, 4, 11]]]
    compare_split_result(result, expected)


def test_split_ls2_coords_segment():
    ls_coords = np.array([[1, 2, 10], [3, 4, 11]])
    result = split_ls_coords(ls_coords, ls_coords[:, 2], [10.5])
    expected = [[[1, 2, 10], [2, 3, 10.5]], [[2, 3, 10.5], [3, 4, 11]]]
    compare_split_result(result, expected)


def test_split_ls3_coords_vertex():
    ls_coords = np.array([[1, 2, 10], [3, 4, 11], [5, 6, 15]])
    result = split_ls_coords(ls_coords, ls_coords[:, 2], [11])
    expected = [[[1, 2, 10], [3, 4, 11]], [[3, 4, 11], [5, 6, 15]]]
    compare_split_result(result, expected)


def test_split_ls3_coords_segment():
    ls_coords = np.array([[1, 2, 10], [3, 4, 11], [5, 6, 21]])
    result = split_ls_coords(ls_coords, ls_coords[:, 2], [16])
    expected = [[[1, 2, 10], [3, 4, 11], [4, 5, 16]], [[4, 5, 16], [5, 6, 21]]]
    compare_split_result(result, expected)


def test_split_ls_large_many():
    ls_dists = np.arange(1000) ** 2 - 100.0
    ls_coords = np.array([[d, -2 * d + 5] for d in ls_dists])
    split_dists = [
        ls_dists[0],
        ls_dists[10],
        (ls_dists[113] + ls_dists[114]) / 2,
        (ls_dists[113] + 2 * ls_dists[114]) / 3,
        (ls_dists[-2] + ls_dists[-1]) / 2,
        ls_dists[-1],
    ]
    result = split_ls_coords(ls_coords, ls_dists, split_dists)
    expected = [
        [ls_coords[0], ls_coords[0]],
        ls_coords[:11],
        list(ls_coords[10:114]) + [(ls_coords[113] + ls_coords[114]) / 2],
        [
            (ls_coords[113] + ls_coords[114]) / 2,
            (ls_coords[113] + 2 * ls_coords[114]) / 3,
        ],
        [(ls_coords[113] + 2 * ls_coords[114]) / 3]
        + list(ls_coords[114:-1])
        + [(ls_coords[-2] + ls_coords[-1]) / 2],
        [(ls_coords[-2] + ls_coords[-1]) / 2, ls_coords[-1]],
        [ls_coords[-1], ls_coords[-1]],
    ]
    compare_split_result(result, expected)


def test_split_shape_all_trusted(simple_ls):
    shape = GeomWithDists(
        simple_ls,
        np.array([10, 100, 1000, 1200], dtype=float),
        np.ones(4, dtype=bool),
        np.ones(4, dtype=bool),
        4,
    )
    # point coords shall not matter, only dists
    points = GeomWithDists(
        MultiPoint([[-5, -5], [-5, -5]]),
        np.array([55, 1100]),
        np.ones(2, dtype=bool),
        np.ones(2, dtype=bool),
        2,
    )
    line_strings = split_shape(shape, points)
    assert len(line_strings) == 1
    assert_allclose(
        np.array(line_strings[0].coords),
        np.array([[5e-4, 0], [1e-3, 0], [1e-3, 2e-3], [3e-3, 2e-3]]),
    )


def test_split_shape_all_but_one_trusted(simple_ls):
    shape = GeomWithDists(
        simple_ls,
        np.array([10, 100, 1000, 1200], dtype=float),
        np.ones(4, dtype=bool),
        np.ones(4, dtype=bool),
        4,
    )
    # coords for point with untrusted dist
    untrusted_coords = [1e-3, 1e-3]
    points = GeomWithDists(
        MultiPoint([[-5, -5], untrusted_coords, [-5, -5]]),
        np.array([55, 500, 1100]),
        np.array([True, False, True]),
        np.ones(3, dtype=bool),
        3,
    )
    line_strings = split_shape(shape, points)
    assert len(line_strings) == 2
    assert_allclose(
        np.array(line_strings[0].coords),
        np.array([[5e-4, 0], [1e-3, 0], untrusted_coords]),
    )
    assert_allclose(
        np.array(line_strings[1].coords),
        np.array([untrusted_coords, [1e-3, 2e-3], [3e-3, 2e-3]]),
    )


@pytest.mark.parametrize("reverse", (True, False))
def test_split_shape_coords_only(simple_ls, reverse):
    if reverse:
        simple_ls = LineString(simple_ls.coords[::-1])
    shape = GeomWithDists(
        simple_ls,
        np.array([None] * 4, dtype=float),
        np.zeros(4, dtype=bool),
        np.zeros(4, dtype=bool),
        0,
    )
    mp = MultiPoint([[1e-4, 5e-4], [6e-3, 3e-3]])
    points = GeomWithDists(
        mp,
        np.array([None, None], dtype=float),
        np.zeros(2, dtype=bool),
        np.zeros(2, dtype=bool),
        0,
    )
    line_strings = split_shape(shape, points)
    assert len(line_strings) == 1
    assert_allclose(
        np.array(line_strings[0].coords),
        np.array([[1e-4, 0], [1e-3, 0], [1e-3, 2e-3], [5e-3, 2e-3]]),
    )


def test_split_shape_wrong_order_one_untrusted(simple_ls):
    shape = GeomWithDists(
        simple_ls,
        np.array([10, 100, 1000, 1200], dtype=float),
        np.ones(4, dtype=bool),
        np.ones(4, dtype=bool),
        4,
    )
    # we reverse the coords but give a hint for the location of one of the points
    # the solution should shift the points close together (but it cannot swap them)
    mp = MultiPoint([[1e-4, 5e-4], [6e-3, 3e-3]][::-1])
    points = GeomWithDists(
        mp,
        np.array([1010, None], dtype=float),
        np.zeros(2, dtype=bool),
        np.array([True, False]),
        1,
    )
    min_dist = 25
    line_strings = split_shape(shape, points, min_dist_meters=min_dist)
    assert len(line_strings) == 1
    ls = line_strings[0]
    assert get_geodesic_line_string_dists(np.array(ls.coords))[-1] == pytest.approx(
        min_dist
    ), "maximally close"
    assert ls.coords[0][0] < ls.coords[-1][0], "correct order"


def test_split_shape_messed_up_trusted_dists(straight_ls, straight_ls_dists):
    shape = GeomWithDists(
        straight_ls,
        straight_ls_dists,
        np.ones(2, dtype=bool),
        np.ones(2, dtype=bool),
        2,
    )
    points = GeomWithDists(
        MultiPoint([[1e-3, -1e-5], [2e-3, 1e-5], [4e-3, 2e-6], [5e-3, 0]]),
        # * -10 is out of bounds, should be moved to 0 and still count as trusted
        # * 500 and 490 are too close together and not increasing, they should end up
        #   symmetrically around 495 with equidistant minimal spacing and count as untrusted
        #   (they move 17.5 meters and only 15 meters will be allowed)
        # * 1200 is out of bounds, should be moved to 1000 and marked as untrusted
        np.array([-10, 500, 490, 1200]),
        np.ones(4, dtype=bool),
        np.ones(4, dtype=bool),
        4,
    )
    line_strings = split_shape(
        shape, points, min_dist_meters=25, max_move_trusted_meters=15
    )
    assert len(line_strings) == 3
    assert_allclose(
        np.array(line_strings[0].coords),
        np.array([[0, 0], [2e-3, 0]]),
        atol=1e-7,
        rtol=0,
    )
    assert_allclose(
        np.array(line_strings[1].coords),
        np.array([[2e-3, 0], [4e-3, 0]]),
        atol=1e-7,
        rtol=0,
    )
    assert_allclose(
        np.array(line_strings[2].coords),
        np.array([[4e-3, 0], [5e-3, 0]]),
        atol=1e-7,
        rtol=0,
    )


def test_split_shape_fill_missing_dists():
    # Simply projecting the points results in the wrong order.
    #
    # x----<-----x
    # 3   1    2 |
    #            ^
    #            |
    # x---->-----x
    # 0

    ls = LineString([[0, 0], [1e-2, 0], [1e-2, 1e-3], [0, 1e-3]])
    shape = GeomWithDists(
        ls,
        np.array([0, 1000, 1100, 2100], dtype=float),
        np.ones(4, dtype=bool),
        np.ones(4, dtype=bool),
        4,
    )
    points = GeomWithDists(
        MultiPoint([[0, -1e-5], [1e-3, 8e-4], [7e-3, 8e-4], [0, 9e-4]]),
        np.array([None] * 4, dtype=float),
        np.zeros(4, dtype=bool),
        np.zeros(4, dtype=bool),
        0,
    )

    # If we don't pass travel times as a hint, dists are filled
    # equidistantly spanning the whole shape.
    # This should put 1 on the lower branch and 2 on the upper branch.
    line_strings = split_shape(shape, points, try_reverse=False)
    assert len(line_strings) == 3
    assert_allclose(
        np.array(line_strings[0].coords),
        np.array([[0, 0], [1e-3, 0]]),
        atol=1e-7,
        rtol=0,
    )
    assert_allclose(
        np.array(line_strings[1].coords),
        np.array([[1e-3, 0], [1e-2, 0], [1e-2, 1e-3], [7e-3, 1e-3]]),
        atol=1e-7,
        rtol=0,
    )
    assert_allclose(
        np.array(line_strings[2].coords),
        np.array([[7e-3, 1e-3], [0, 1e-3]]),
        atol=1e-7,
        rtol=0,
    )

    # Now we pass travel times indicating that both 1 and 2 should
    # be on the upper branch.
    travel_seconds = [7 * 60, 0, 3 * 60]
    line_strings = split_shape(
        shape,
        points,
        travel_seconds,
        try_reverse=False,
        # tolerate snapped stop ending up far away from original stop
        res_atol=1000,
    )
    assert len(line_strings) == 3

    # from output and this makes sense
    x1_expected = 0.00641
    x2_expected = 0.006186

    assert_allclose(
        np.array(line_strings[0].coords),
        np.array([[0, 0], [1e-2, 0], [1e-2, 1e-3], [x1_expected, 1e-3]]),
        atol=1e-6,
        rtol=0,
    )
    assert_allclose(
        np.array(line_strings[1].coords),
        np.array([[x1_expected, 1e-3], [x2_expected, 1e-3]]),
        atol=1e-6,
        rtol=0,
    )
    assert_allclose(
        np.array(line_strings[2].coords),
        np.array([[x2_expected, 1e-3], [0, 1e-3]]),
        atol=1e-6,
        rtol=0,
    )


_many_d = [0, 25, 80] + list(np.arange(400, 700, 25)) + [900, 998]


@pytest.mark.parametrize(
    "mp,d_mp,atol,expected",
    [
        # degenerate case
        (MultiPoint([]), [], None, []),
        # one point
        (MultiPoint([[2.1e-3, 1e-4]]), [500], None, [210]),
        # two points
        (MultiPoint([[5e-3, 1e-4], [6e-3, -1e-4]]), [620, 650], None, [500, 600]),
        # two points with snapped coords in wrong order
        (MultiPoint([[8e-3, 1e-4], [7e-3, -1e-4]]), [100, 900], None, [700 - 25, 700]),
        # two points with snapped coords in wrong order with better dist hints
        # (same result, but should converge faster)
        (MultiPoint([[8e-3, 1e-4], [7e-3, -1e-4]]), [650, 690], None, [700 - 25, 700]),
        # many unevenly spaced points without equidistant dist hint (will it converge?)
        (
            MultiPoint([[d * 1e-5, 1e-4] for d in _many_d]),
            np.linspace(0, 1000, len(_many_d)),
            None,
            _many_d,
        ),
        # exactly as many points as we can fit, all with the same coords
        # optimization should terminate with the initial equidistant dists
        # but we need to increase atol, so the solution is accepted (this is
        # bad data that should not occur in reality)
        (
            MultiPoint([[0, 0] for _ in range(41)]),
            np.linspace(0, 1000, 41),
            1200,
            np.linspace(0, 1000, 41),
        ),
        # same as above but with default atol shall give an error
        (
            MultiPoint([[0, 0] for _ in range(41)]),
            np.linspace(0, 1000, 41),
            None,
            SnappingError,
        ),
        # too many points
        (
            MultiPoint([[0, 0] for _ in range(42)]),
            np.linspace(0, 1000, 42),
            None,
            SnappingError,
        ),
    ],
)
def test_snap(straight_ls, straight_ls_dists, mp, d_mp, atol, expected):
    d_mp = np.asarray(d_mp, dtype=float)
    if atol is not None:
        atol_snap = partial(snap, res_atol=atol)
    else:
        atol_snap = snap
    if type(expected) == type and issubclass(expected, Exception):
        with pytest.raises(expected):
            atol_snap(straight_ls, straight_ls_dists, mp, d_mp)
    else:
        result, reverse = atol_snap(straight_ls, straight_ls_dists, mp, d_mp)
        assert not reverse
        # result should be converged to about 1 meter
        assert_allclose(result, expected, rtol=0, atol=1.1)
