from functools import partial
from itertools import product

import numpy as np
from numpy import array as arr
from numpy.testing import assert_allclose
import pytest

from pysnapping import ExtrapolationError
from pysnapping.linear_referencing import (
    locate,
    Locations,
    resample,
    substrings,
    ProjectionTarget,
    location_to_single_segment,
)


@pytest.fixture
def simple_ls():
    r"""A simple linestring with 3 segments and 4 vertices

       v1____v2
       /  s1  \
    s0/        \s2
     /          \
    v0          v3
    """
    return arr([[0, 0], [1, 1], [2, 1], [3, 0]], dtype=float)


def assert_locations_eq(l1, l2):
    assert np.all(l1.from_vertices == l2.from_vertices)
    assert np.all(l1.to_vertices == l2.to_vertices)
    assert_allclose(l1.fractions, l2.fractions, rtol=0, atol=1e-8)


def assert_loc(where, what, expected_from, expected_to, expected_fractions, **kwargs):
    if isinstance(what, (float, int)):
        what = [what]
        expected_from = [expected_from]
        expected_to = [expected_to]
        expected_fractions = [expected_fractions]
    assert_locations_eq(
        locate(arr(where), arr(what), **kwargs),
        Locations(arr(expected_from), arr(expected_to), arr(expected_fractions)),
    )


def make_locs(from_to_fractions):
    data = np.array(from_to_fractions)
    return Locations(
        data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2].astype(float)
    )


def make_loc(i1, i2, f):
    return make_locs([(i1, i2, f)])


def test_location_to_single_segment():
    fun = partial(location_to_single_segment, last_segment=10)
    assert fun(1, 1, 0.5, prefer_zero_fraction=True) == (1, 0.0)
    assert fun(10, 11, 1.0, prefer_zero_fraction=True) == (10, 1.0)
    assert fun(1, 1, 0.5, prefer_zero_fraction=False) == (0, 1.0)
    assert fun(0, 1, 0.0, prefer_zero_fraction=False) == (0, 0.0)
    with pytest.raises(ExtrapolationError):
        fun(0, 0, -0.1, prefer_zero_fraction=True)
    with pytest.raises(ExtrapolationError):
        fun(0, 0, 1.1, prefer_zero_fraction=True)
    with pytest.raises(ExtrapolationError):
        fun(1, 3, -0.1, prefer_zero_fraction=True)
    with pytest.raises(ExtrapolationError):
        fun(1, 3, 1.1, prefer_zero_fraction=True)
    assert fun(0, 2, 0.0, prefer_zero_fraction=False) == (0, 0.0)
    assert fun(0, 2, 1.0, prefer_zero_fraction=True) == (2, 0.0)
    assert fun(0, 2, 0.5, prefer_zero_fraction=True) == (1, 0.0)
    assert fun(0, 2, 0.5, prefer_zero_fraction=False) == (0, 1.0)
    assert fun(11, 11, 0.5, prefer_zero_fraction=False) == (10, 1.0)
    assert fun(11, 11, 0.5, prefer_zero_fraction=True) == (10, 1.0)


def test_locations_order():
    ls = [
        [(1, 2, 0.5), (1, 2, 0.4)],
        [(1, 2, 1), (3, 3, 0)],
        [(1, 3, 0.5), (0, 2, 0.5)],
    ]
    expected_maxs = [0, 1, 0]
    l1 = make_locs([item[0] for item in ls])
    l2 = make_locs([item[1] for item in ls])
    l_exp = make_locs([item[i] for item, i in zip(ls, expected_maxs)])
    assert_locations_eq(l1.max(l2), l_exp)
    assert_locations_eq(l2.max(l1), l_exp)


def test_locate():
    assert_loc(
        [1, 2, 4],
        [2.5, 1, 4, 2, 0.99, 4.01],
        [1, 0, 2, 1, 0, 2],
        [2, 0, 2, 1, 0, 2],
        [0.25, 0.5, 0.5, 0.5, 0, 1],
    )
    assert_loc([1, 1, 2, 4], 1, 0, 1, 0.5)
    assert_loc([1, 2, 4, 4], 4, 2, 3, 0.5)
    assert_loc([1, 2, 4, 4, 5], 4, 2, 3, 0.5)
    assert_loc([1, 2, 4, 4, 4, 5], 4, 2, 4, 0.5)
    assert_loc([1, 2, 4, 4, 4], 4, 2, 4, 0.5)
    assert_loc([1, 1, 1, 2, 4], 1, 0, 2, 0.5)
    assert_loc([0, 1, 1, 1, 2, 4], 1, 1, 3, 0.5)
    assert_loc([0, 1, 1, 1, 1, 2, 4], 1, 1, 4, 0.5)
    assert_loc([1, 1, 1, 1, 2, 4], 1, 0, 3, 0.5)
    assert_loc([1, 2, 4, 4, 4, 4], 4, 2, 5, 0.5)
    assert_loc([1, 2, 4, 4, 4, 4, 5], 4, 2, 5, 0.5)
    assert_loc([1, 1, 2, 4], 0.99, 0, 0, 0)
    assert_loc([1, 2, 4, 4], 4.01, 3, 3, 1)
    assert_loc([1, 1, 1, 2, 4], 0.99, 0, 0, 0)
    assert_loc([1, 2, 4, 4, 4], 4.01, 4, 4, 1)
    assert_loc([1, 1], 1, 0, 1, 0.5)
    assert_loc([1, 1, 1], 1, 0, 2, 0.5)
    assert_loc([1, 1], [], [], [], [])
    assert_loc([1, 2, 4], [0.5, 4.1], [0, 1], [1, 2], [-0.5, 1.05], extrapolate=True)
    assert_loc([1, 1, 2, 4], 0.5, 0, 2, -0.5, extrapolate=True)
    assert_loc([1, 2, 4, 4], 4.1, 1, 3, 1.05, extrapolate=True)
    with pytest.raises(ExtrapolationError):
        locate(arr([1, 1]), arr([1, 0.9]), True)
    with pytest.raises(ExtrapolationError):
        locate(arr([1, 1, 1]), arr([1.1]), True)


def test_resample():
    x = arr([1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 4.0])
    y = arr([2.0, 2.2, 3.0, 3.2, 4.0, 1000, 4.2])
    x_prime = arr([-0.5, 1.0, 1.25, 2.0, 3.0, 3.5, 4.0, 6.0])
    y_prime = arr([+2.0, 2.1, 2.40, 3.1, 3.6, 3.8, 4.1, 4.2])
    assert_allclose(resample(x, y, x_prime), y_prime, rtol=0, atol=1e-8)

    y_prime[0] = 2 + (-0.5 - 1.0) / (2.0 - 1.0) * (3 - 2)
    y_prime[-1] = 3.2 + (6 - 2) / (4 - 2) * (4.2 - 3.2)
    assert_allclose(resample(x, y, x_prime, True), y_prime, rtol=0, atol=1e-8)

    # y with higher dimension
    x = arr([0.0, 1.0])
    y = arr([[1.0, 2, 3], [4, 5, 6]])
    x_prime = arr([0.5, 1.1])
    y_prime = arr([[2.5, 3.5, 4.5], [4, 5, 6]])
    assert_allclose(resample(x, y, x_prime), y_prime, rtol=0, atol=1e-8)

    # nan stays nan but does not harm the other values
    x = arr([1.0, 2])
    y = arr([1.0, 2])
    x_prime = arr([1, 1.5, 2, None], dtype=float)
    y_prime = arr([1, 1.5, 2, None], dtype=float)
    assert_allclose(resample(x, y, x_prime), y_prime, rtol=0, atol=1e-8, equal_nan=True)
    x = arr([1.0, 2])
    y = arr([[1.0, 1.0], [2, 2]])
    x_prime = arr([1, 1.5, 2, None], dtype=float)
    y_prime = arr([[1, 1], [1.5, 1.5], [2, 2], [None, None]], dtype=float)
    assert_allclose(resample(x, y, x_prime), y_prime, rtol=0, atol=1e-8, equal_nan=True)


def test_substrings(simple_ls):
    ls = simple_ls
    start1 = make_loc(0, 0, 0.5)
    end1 = make_loc(3, 3, 0.5)
    start2 = make_loc(0, 1, 0)
    end2 = make_loc(2, 3, 1)
    v1 = make_loc(1, 1, 0.5)
    s1_half = make_loc(1, 2, 0.5)  # middle of segment with index 1

    # start to end
    assert_allclose(substrings(ls, start1, end1), [ls])
    assert_allclose(substrings(ls, start2, end2), [ls])
    assert_allclose(substrings(ls, start1, end2), [ls])
    assert_allclose(substrings(ls, start2, end1), [ls])

    # start to start
    assert_allclose(substrings(ls, start1, start1), [ls[[0, 0]]])
    assert_allclose(substrings(ls, start2, start2), [ls[[0, 0]]])
    assert_allclose(substrings(ls, start1, start2), [ls[[0, 0]]])
    assert_allclose(substrings(ls, start2, start1), [ls[[0, 0]]])

    # end to end
    assert_allclose(substrings(ls, end1, end1), [ls[[3, 3]]])
    assert_allclose(substrings(ls, end2, end2), [ls[[3, 3]]])
    assert_allclose(substrings(ls, end1, end2), [ls[[3, 3]]])
    assert_allclose(substrings(ls, end2, end1), [ls[[3, 3]]])

    # vertex 1 to vertex1
    assert_allclose(substrings(ls, v1, v1), [ls[[1, 1]]])

    # in segment to same
    assert_allclose(substrings(ls, s1_half, s1_half), [[[1.5, 1], [1.5, 1]]])

    # fraction of 1 segment
    assert_allclose(
        substrings(ls, make_loc(1, 2, 0.25), make_loc(1, 2, 0.75)),
        [[[1.25, 1], [1.75, 1]]],
    )

    # fraction of 1 segment with start/end on vertex
    assert_allclose(
        substrings(ls, make_loc(1, 1, 0.5), make_loc(1, 2, 0.75)), [[[1, 1], [1.75, 1]]]
    )
    assert_allclose(
        substrings(ls, make_loc(1, 2, 0.25), make_loc(1, 2, 1)), [[[1.25, 1], [2, 1]]]
    )

    # middle of segment to middle of segment with another segment in between
    assert_allclose(
        substrings(ls, make_loc(0, 1, 0.5), make_loc(2, 3, 0.5)),
        [[[0.5, 0.5], ls[1], ls[2], [2.5, 0.5]]],
    )

    # end before start (shall be start to start)
    assert_allclose(
        substrings(ls, make_loc(1, 1, 0.5), make_loc(0, 1, 0.25)), [ls[[1, 1]]]
    )

    # extrapolate start
    assert_allclose(substrings(ls, make_loc(0, 1, -1), v1), [[[-1, -1], ls[0], ls[1]]])

    # extrapolate end
    assert_allclose(
        substrings(ls, v1, make_loc(2, 3, 1.5)), [[ls[1], ls[2], ls[3], [3.5, -0.5]]]
    )

    # extrapolate middle
    assert_allclose(
        substrings(ls, make_loc(1, 2, -1), make_loc(1, 2, 2)),
        [[[0, 1], ls[1], ls[2], [3, 1]]],
    )
    assert_allclose(substrings(ls, make_loc(1, 2, -1), v1), [[[0, 1], ls[1]]])

    # multiple at once
    assert_allclose(
        substrings(
            ls,
            make_locs([(0, 1, 0.5), (0, 0, 0.5)]),
            make_locs([(1, 1, 0.5), (0, 1, -1)]),
        ),
        [[[0.5, 0.5], ls[1]], ls[[0, 0]]],
    )


def test_project_to_all_segments(simple_ls):
    points = arr([[-0.5, -0.4], simple_ls[1], [2, 0], simple_ls[3]])

    target = ProjectionTarget(simple_ls)
    projected = target.project(points)

    expected_locations = make_locs(
        [
            (0, 1, 0),
            (0, 1, 1),
            (2, 3, 0.5),
            (2, 3, 1),
        ]
    )
    expected_proj_points = arr([simple_ls[0], simple_ls[1], [2.5, 0.5], simple_ls[3]])
    expected_distances = arr([np.linalg.norm(points[0]), 0, 0.5 * 2**0.5, 0])

    assert_locations_eq(projected.locations, expected_locations)
    assert_allclose(projected.coords, expected_proj_points)
    assert_allclose(projected.cartesian_distances, expected_distances)

    # now with infinite head/tail
    projected = target.project(points, head_fraction=-np.inf, tail_fraction=np.inf)

    # project point 0 with pen and paper to get the expected result:
    # (all other points are not affected)
    expected_distances[0] = 0.1 * np.sin(45 / 180 * np.pi)
    expected_locations.fractions[0] = (
        -0.5 * 2**0.5 + expected_distances[0]
    ) / 2**0.5
    expected_proj_points[0] = expected_locations.fractions[0]

    assert_locations_eq(projected.locations, expected_locations)
    assert_allclose(projected.coords, expected_proj_points)
    assert_allclose(projected.cartesian_distances, expected_distances)

    # make sure it also works at the end by reversing the linestring:
    target = ProjectionTarget(simple_ls[::-1])
    projected = target.project(points, head_fraction=-np.inf, tail_fraction=np.inf)

    expected_fraction = 1 - expected_locations.fractions[0]

    assert_allclose(projected.locations.fractions[0], expected_fraction)
    assert_allclose(projected.locations.from_vertices[0], 2)
    assert_allclose(projected.locations.to_vertices[0], 3)

    assert_allclose(projected.coords, expected_proj_points)
    assert_allclose(projected.cartesian_distances, expected_distances)


def test_project_to_substring(simple_ls):
    target = ProjectionTarget(simple_ls)
    points = arr([simple_ls[0], simple_ls[3]])
    distances = arr([0, 1, 2, 3], dtype=float)
    lfracs = target.get_line_fractions(points)

    # compare to first creating a substring and then projecting
    distance_pool = np.arange(-2.5, 5.6, 0.5)
    extrapolate_pool = [True, False]
    for d1, d2, extrapolate in product(distance_pool, distance_pool, extrapolate_pool):
        result = lfracs.project_between_distances(d1, d2, distances, extrapolate)
        sub_target = ProjectionTarget(
            substrings(
                simple_ls,
                locate(distances, arr([d1]), extrapolate),
                locate(distances, arr([d2]), extrapolate),
            )[0]
        )
        expected_result = sub_target.project(points)

        # fractions and segments are shifted when first taking a substring, so we only compare
        # coords and distances
        msg = f"works for {d1}, {d2}, {extrapolate}"
        assert_allclose(
            result.cartesian_distances, expected_result.cartesian_distances
        ), msg
        assert_allclose(result.coords, expected_result.coords), msg

    # explicit test to also check segment indices and fractions
    d1, d2 = 1.6, 2.3
    result = lfracs.project_between_distances(d1, d2, distances)
    assert_locations_eq(result.locations, make_locs([(1, 2, 0.6), (2, 3, 0.3)]))

    # test point slicing
    with pytest.raises(KeyError):
        lfracs[0]
    with pytest.raises(KeyError):
        lfracs[:, :]
    sliced_lfracs = lfracs[[0, 0]]
    result = sliced_lfracs.project_between_distances(d1, d2, distances)
    assert_locations_eq(result.locations, make_locs([(1, 2, 0.6), (1, 2, 0.6)]))
