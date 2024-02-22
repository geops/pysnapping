import typing
import numpy as np
from numpy.testing import assert_allclose
import pytest
import pyproj
from itertools import product
from functools import partial
import os
import json

from pysnapping.snap import (
    DubiousTrajectory,
    DubiousTrajectoryTrip,
    SnappingMethod,
    SnappingParams,
)
from pysnapping.linear_referencing import locate, interpolate, resample
from pysnapping.util import get_trafo, transform_coords
from pysnapping import EPSG4326, EPSG4978


if typing.TYPE_CHECKING:
    from numpy.typing import ArrayLike


WGS84_GEOD = pyproj.Geod(ellps="WGS84")


TO_EPSG4326 = partial(
    transform_coords, trafo=get_trafo(from_crs=EPSG4978, to_crs=EPSG4326)
)


def offset(point, dist, azimuth):
    return np.array(WGS84_GEOD.fwd(*point, az=azimuth, dist=dist)[:2], dtype=float)


def make_dubious_traj(real_segment_lengths, azimuths, dubious_dists, start=(7, 40)):
    points = [start]
    for dist, az in zip(real_segment_lengths, azimuths):
        points.append(offset(points[-1], dist, az))
    # default z is 0 (we call from_xyd_and_z to cover it in tests)
    return DubiousTrajectory.from_xyd_and_z(
        [[p[0], p[1], d] for p, d in zip(points, dubious_dists)]
    )


def segment_lengths_to_dists(segment_lengths, start=0):
    dists = np.full((len(segment_lengths) + 1,), start)
    dists[1:] += np.cumsum(segment_lengths)
    return dists


@pytest.mark.parametrize(
    "values,d_min,d_max,min_spacing,ok",
    [
        ([], 0, 0, 10, True),
        ([np.NaN], 0, 0, 10, True),
        ([0], 0, 0, 10, True),
        ([1], 0, 1, 10, True),
        ([1], 0, 1 - 1e-6, 10, False),
        ([np.NaN, 1, np.NaN], 0, 2, 1 - 1e-6, True),
        ([np.NaN, 1, np.NaN], 0, 1.9, 1, False),
        ([-1, np.NaN, np.NaN, 0], -10, 10, 1, False),
        ([-1, np.NaN, np.NaN, 0], -10, 10, 0.5, False),
        ([-1, np.NaN, np.NaN, 0], -10, 10, 1 / 3 - 1e-6, True),
        ([np.NaN, np.NaN, np.NaN], -100, 0, 50 - 1e-6, True),
        ([np.NaN, np.NaN, np.NaN], -100, 0, 50 + 1e-6, False),
    ],
)
def test_spacing_ok(
    values: "ArrayLike", d_min: float, d_max: float, min_spacing: float, ok: bool
) -> None:
    params = SnappingParams(min_spacing=min_spacing)
    assert params.spacing_ok(values, d_min, d_max) == ok


# test around the entire earth with different orientations
@pytest.mark.parametrize(
    "lon,lat,azimuth",
    list(
        product(
            np.linspace(-180, 180, 5), np.linspace(-90, 90, 5), np.linspace(0, 360, 5)
        )
    ),
)
def test_long_segment_accuracy(lon, lat, azimuth):
    # one segment with 100 km geodesic length
    real_length = 100_000
    dtraj = make_dubious_traj([real_length], [azimuth], [0, 1], start=(lon, lat))
    # we use the trip only to get the converted trajectory
    dtrip = DubiousTrajectoryTrip(dtraj, np.zeros((2, 4)))
    trip = dtrip.to_trajectory_trip()
    # 1.1 meter deviation for 100 km segment is really more than we need
    # since typically segments in public transport data are much shorter
    assert trip.trajectory.length == pytest.approx(real_length, rel=0, abs=1.1)


def test_convert_good_dists():
    real_segment_lengths = [100, 1500, 200, 26]
    dtraj = make_dubious_traj(
        real_segment_lengths, [0, 15, -15, 20], [10, 20, 30, 40, 50]
    )
    dubious_trip_dists = np.array([15, 37.5])
    coords = interpolate(dtraj.xy, locate(dtraj.dists, dubious_trip_dists))
    trip_xyzd = [
        tuple(offset(c, 30, 90)) + (0, d) for d, c in zip(dubious_trip_dists, coords)
    ]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzd=trip_xyzd,
    )
    trip = dtrip.to_trajectory_trip()

    # compare EPSG4978 length to geodesic length
    assert trip.trajectory.length == pytest.approx(1826)
    assert_allclose(
        trip.trajectory.dists, segment_lengths_to_dists(real_segment_lengths)
    )

    # 15 is halfway between 10 and 20, thus halfway of 0->100 = 50
    # 37.5 is 3/4 of 30->40 thus 3/4 of 1600 -> 1800 = 1750
    assert_allclose(trip.dists, [50, 1750])


def test_snap_simple_forward():
    length = 1000
    dtraj = make_dubious_traj(
        [length],
        [0],
        [None, None],
    )
    trip_xyzd = np.full((2, 4), np.nan)
    trip_xyzd[:, :3] = dtraj.xyz
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzd=trip_xyzd,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert not snapped.reverse_order
    assert np.all(snapped.methods == SnappingMethod.routed)
    assert_allclose(snapped.snapping_distances, 0)
    assert_allclose(snapped.shortest_distances, 0)
    assert_allclose(
        TO_EPSG4326(snapped.snapped_points.coords), dtrip.xyz, rtol=0, atol=1e-6
    )
    assert_allclose(snapped.distances, [0, 1000])
    assert_allclose(
        snapped.get_inter_point_ls_lon_lat_in_travel_direction(), [dtraj.xy]
    )


def test_snap_simple_reversed():
    length = 1000
    dtraj = make_dubious_traj(
        [length],
        [0],
        [None, None],
    )
    trip_xyzd = np.full((2, 4), np.nan)
    trip_xyzd[:, :3] = dtraj.xyz[::-1]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzd=trip_xyzd,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert snapped.reverse_order
    assert np.all(snapped.methods == SnappingMethod.routed)
    assert_allclose(snapped.snapping_distances, 0)
    assert_allclose(snapped.shortest_distances, 0)
    assert_allclose(
        TO_EPSG4326(snapped.snapped_points.coords), dtrip.xyz, rtol=0, atol=1e-6
    )
    assert_allclose(snapped.distances, [1000, 0])
    assert_allclose(
        snapped.get_inter_point_ls_lon_lat_in_travel_direction(), [dtraj.xy[::-1]]
    )


def test_snap_nontrivial_forward():
    params = SnappingParams(min_spacing=20, sampling_step=0.5)
    length = 1000
    dtraj = make_dubious_traj(
        [length],
        [0],
        [None, None],
    )
    # put point 0 at 0 meters, point 1 at 1000 meters and point 2 at 900 meters so the
    # order is messed up
    trip_xyzd = np.full((3, 4), np.nan)
    trip_xyzd[:2, :3] = dtraj.xyz
    real_dists = np.array([0, 1000.0])
    trip_xyzd[2:3, :3] = resample(real_dists, dtraj.xyz, np.array([900.0]))
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzd=trip_xyzd,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points(params)
    assert not snapped.reverse_order
    assert np.all(snapped.methods == SnappingMethod.routed)
    assert_allclose(snapped.shortest_distances, 0, rtol=0, atol=1)
    # sum of square distances is minimized for the symmetric solution that respectes the
    # minimum spacing
    expected_dists = np.array([0, 940, 960], dtype=float)
    assert_allclose(snapped.distances, expected_dists, rtol=0, atol=1)
    assert_allclose(snapped.snapping_distances, [0, 60, 60], rtol=0, atol=1)
    expected_lon_lat = resample(real_dists, dtraj.xy, expected_dists)
    assert_allclose(
        TO_EPSG4326(snapped.snapped_points.coords, skip_z_output=True),
        expected_lon_lat,
        rtol=0,
        atol=1e-5,
    )
    assert_allclose(
        snapped.get_inter_point_ls_lon_lat_in_travel_direction(),
        [expected_lon_lat[[0, 1]], expected_lon_lat[[1, 2]]],
        rtol=0,
        atol=1e-5,
    )


def test_snap_iterative_backward():
    params = SnappingParams(min_spacing=20, sampling_step=0.5)
    length = 1000
    dtraj = make_dubious_traj(
        [length],
        [0],
        [None, None],
    )
    # put point 0 at 1000 meters, point 1 at 0 meters and point 2 at 500 meters so the
    # order is messed up and the reverse solution is preferred
    trip_xyzd = np.full((3, 4), np.nan)
    trip_xyzd[:2, :3] = dtraj.xyz[::-1]
    real_dists = np.array([0, 1000.0])
    trip_xyzd[2:3, :3] = resample(real_dists, dtraj.xyz, np.array([500.0]))
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzd=trip_xyzd,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points(params)
    assert snapped.reverse_order
    assert np.all(snapped.methods == SnappingMethod.routed)
    assert_allclose(snapped.shortest_distances, 0, rtol=0, atol=1)
    # sum of square distances is minimized for the symmetric solution that respectes the
    # minimum spacing
    expected_dists = np.array([1000.0, 260, 240])
    assert_allclose(snapped.distances, expected_dists, rtol=0, atol=1)
    assert_allclose(snapped.snapping_distances, [0, 260, 260], rtol=0, atol=1)
    expected_lon_lat = resample(real_dists, dtraj.xy, expected_dists)
    assert_allclose(
        TO_EPSG4326(snapped.snapped_points.coords, skip_z_output=True),
        expected_lon_lat,
        rtol=0,
        atol=1e-5,
    )
    assert_allclose(
        snapped.get_inter_point_ls_lon_lat_in_travel_direction(),
        [expected_lon_lat[[0, 1]], expected_lon_lat[[1, 2]]],
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("seed", range(20))
def test_short_trajectory_and_random_garbage(seed):
    # set parameters such that a solution is always possible
    params = SnappingParams(
        # tolerate arbitrary distance of points to trajectory
        max_shortest_distance=float("inf"),
        # tolerate arbitrary distance of points to snapped points
        rtol_snap=0,
        atol_snap=float("inf"),
    )

    rng = np.random.default_rng(seed=seed)

    if seed == 0:
        # special case: short trajectory with many points
        n_segments = 1
        segment_lengths = np.array([100.0])
        n_points = 50
        z = 0
    elif seed == 1:
        # special case: zero length trajectory ("elevator" in 2d)
        n_segments = 1
        segment_lengths = np.array([0.0])
        n_points = 3
        z = 0
    else:
        n_segments = rng.integers(1, 1001)
        segment_lengths = rng.uniform(0, 3000, n_segments)
        n_points = rng.integers(2, 51)
        z = np.cumsum(rng.uniform(-10, 10, n_points))

    azimuths = rng.uniform(0, 360, n_segments)
    dtraj_dists = np.linspace(0, 1000, n_segments + 1)
    dtraj = make_dubious_traj(segment_lengths, azimuths, dtraj_dists)

    trip_xyzd = np.empty((n_points, 4))
    trip_xyzd[:, 0] = rng.uniform(6.9, 7.1, n_points)
    trip_xyzd[:, 1] = rng.uniform(39.9, 40.1, n_points)
    trip_xyzd[:, 2] = z
    trip_xyzd[:, 3] = rng.uniform(-100, 1100, n_points)

    # make some distances missing
    n_missing = rng.integers(0, n_points + 1)
    trip_xyzd[rng.choice(n_points, n_missing, replace=False), 3] = np.nan

    dtrip = DubiousTrajectoryTrip(dtraj, trip_xyzd)
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points(params)
    min_spacing = min(params.min_spacing, trip.trajectory.length / (n_points - 1))
    for split_segment in snapped.get_inter_point_ls_lon_lat_in_travel_direction():
        assert (
            WGS84_GEOD.line_length(split_segment[:, 0], split_segment[:, 1])
            >= 0.99 * min_spacing
        )


def test_complex_trip():
    """Hard case that xfailed previously with the iterative solution.

    It is a trip with some detours and no distance information.
    """
    fn = os.path.join(os.path.dirname(__file__), "complex_trip.geojson")
    with open(fn) as handle:
        coll = json.load(handle)
    point_features = []
    for feature in coll["features"]:
        if feature["geometry"]["type"] == "LineString":
            traj_feature = feature
        else:
            point_features.append(feature)
    point_features.sort(key=lambda f: f["properties"]["index"])

    dtraj = DubiousTrajectory(
        [c + [None] for c in traj_feature["geometry"]["coordinates"]]
    )
    dtrip = DubiousTrajectoryTrip(
        dtraj,
        [f["geometry"]["coordinates"] + [None] for f in point_features],
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert not snapped.reverse_order


# 25 is default min_spacing and 5 is default sampling_step.
# The cutoff is at 25 + 2.01 * 5 thus 35 should be too close but 36 should be good.
@pytest.mark.parametrize(
    "spacing,method", ((35, SnappingMethod.routed), (36, SnappingMethod.trusted))
)
def test_untrust_bad_spacing(spacing: float, method: SnappingMethod) -> None:
    length = 50
    dtraj = make_dubious_traj(
        [length],
        [0],
        [0, 50],
    )
    trip_xyzd = np.full((2, 4), np.nan)
    trip_xyzd[:, :3] = dtraj.xyz

    trip_xyzd[:, 3] = [10, 10 + spacing]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzd=trip_xyzd,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert not snapped.reverse_order
    assert np.all(snapped.methods == method)


# point is on trajectory, so only atol_snap matters which is 10 by default
@pytest.mark.parametrize(
    "distance,method", ((9, SnappingMethod.trusted), (11, SnappingMethod.routed))
)
def test_untrust_too_far_away(distance: float, method: SnappingMethod) -> None:
    length = 1000
    dtraj = make_dubious_traj(
        [length],
        [0],
        [0, 1000],
    )
    trip_xyzd = np.full((2, 4), np.nan)
    trip_xyzd[:, :3] = dtraj.xyz

    trip_xyzd[:, 3] = [0, 1000 - distance]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzd=trip_xyzd,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert not snapped.reverse_order
    assert snapped.methods[0] == SnappingMethod.trusted
    assert snapped.methods[1] == method
