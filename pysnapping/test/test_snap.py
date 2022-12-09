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
    dtrip = DubiousTrajectoryTrip(dtraj, np.zeros((2, 5)))
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
    trip_xyzdt = [
        tuple(offset(c, 30, 90)) + (0, d, None)
        for d, c in zip(dubious_trip_dists, coords)
    ]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzdt=trip_xyzdt,
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
    assert np.all(trip.dists_trusted)


def test_convert_using_times_and_dists():
    real_segment_lengths = [100, 1500, 200, 26]
    dtraj = make_dubious_traj(
        real_segment_lengths,
        [0, 15, -15, 20],
        [None, 10, 160, None, None],
    )

    # two distance/time combinations are given, so we can interpolate and extrapolate the rest
    dists = [None, 20, None, 150, None, None]
    times = [300, 400, 2000, 3000, 3040, 4000]

    coords = interpolate(dtraj.xy, locate(dtraj.dists, np.array(dists, dtype=float)))
    trip_xyzdt = [
        (tuple(offset(c, 30, 90)) if d is not None else (7, 40)) + (0, d, t)
        for c, d, t in zip(coords, dists, times)
    ]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzdt=trip_xyzdt,
    )
    trip = dtrip.to_trajectory_trip()
    # values are chosen s.th. times are metric dists x2, so time 300 -> metric dist 150,
    # 400 -> 200 (trusted input),
    # 2000 -> 1000,
    # 3000 -> 1500 (trusted input),
    # 3040 -> 1520 but too close to 1500 and since 1500 is trusted -> 1525 (min spacing 25 meters)
    # and 4000 -> 2000 but trajectory ends at 1826.
    assert_allclose(trip.dists, [150, 200, 1000, 1500, 1525, 1826])
    assert np.all(trip.dists_trusted == [False, True, False, True, False, False])


def test_convert_using_only_times():
    length = 1000
    dtraj = make_dubious_traj(
        [length],
        [0],
        [None, None],
    )

    times = [0, 1000, 2500, 5000]

    trip_xyzdt = [(7, 40, 0, None, t) for t in times]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzdt=trip_xyzdt,
    )
    trip = dtrip.to_trajectory_trip()
    # if no trip point distances are available,
    # it is assumed that the trip spans the whole trajectory (1000 meters in 5000 seconds)
    assert_allclose(trip.dists, [t / 5 for t in times])
    assert not np.any(trip.dists_trusted)


def test_snap_simple_forward():
    length = 1000
    dtraj = make_dubious_traj(
        [length],
        [0],
        [None, None],
    )
    trip_xyzdt = np.full((2, 5), np.nan)
    trip_xyzdt[:, :3] = dtraj.xyz
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzdt=trip_xyzdt,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert not snapped.reverse_order
    assert snapped.methods[0] == SnappingMethod.projected
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
    trip_xyzdt = np.full((2, 5), np.nan)
    trip_xyzdt[:, :3] = dtraj.xyz[::-1]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzdt=trip_xyzdt,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert snapped.reverse_order
    assert snapped.methods[0] == SnappingMethod.projected
    assert_allclose(snapped.snapping_distances, 0)
    assert_allclose(snapped.shortest_distances, 0)
    assert_allclose(
        TO_EPSG4326(snapped.snapped_points.coords), dtrip.xyz, rtol=0, atol=1e-6
    )
    assert_allclose(snapped.distances, [1000, 0])
    assert_allclose(
        snapped.get_inter_point_ls_lon_lat_in_travel_direction(), [dtraj.xy[::-1]]
    )


def test_snap_iterative_forward():
    length = 1000
    dtraj = make_dubious_traj(
        [length],
        [0],
        [None, None],
    )
    trip_xyzdt = np.full((3, 5), np.nan)
    trip_xyzdt[:2, :3] = dtraj.xyz
    real_dists = np.array([0, 1000.0])
    trip_xyzdt[2:3, :3] = resample(real_dists, dtraj.xyz, np.array([900.0]))
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzdt=trip_xyzdt,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert not snapped.reverse_order
    assert snapped.methods[0] == SnappingMethod.iterative
    assert_allclose(snapped.shortest_distances, 0, rtol=0, atol=1)
    # 3 is closer to the initial guess (equidistant placement) and thus ends up at
    # its global otpimum at 900 before 2 has a chance to move from 500 up; min dist is 25 meters
    expected_dists = np.array([0, 875, 900.0])
    assert_allclose(snapped.distances, expected_dists, rtol=0, atol=1)
    assert_allclose(snapped.snapping_distances, [0, 125, 0], rtol=0, atol=1)
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
    length = 1000
    dtraj = make_dubious_traj(
        [length],
        [0],
        [None, None],
    )
    trip_xyzdt = np.full((3, 5), np.nan)
    trip_xyzdt[:2, :3] = dtraj.xyz[::-1]
    real_dists = np.array([0, 1000.0])
    trip_xyzdt[2:3, :3] = resample(real_dists, dtraj.xyz, np.array([500.0]))
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xyzdt=trip_xyzdt,
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert snapped.reverse_order
    assert snapped.methods[0] == SnappingMethod.iterative
    assert_allclose(snapped.shortest_distances, 0, rtol=0, atol=1)
    # initial equidistant placement lets 2 and 3 fight for 250
    # but min spacing is 25 meters
    expected_dists = np.array([1000.0, 250 + 25 / 2, 250 - 25 / 2])
    assert_allclose(snapped.distances, expected_dists, rtol=0, atol=1)
    assert_allclose(
        snapped.snapping_distances, [0, 250 + 25 / 2, 250 + 25 / 2], rtol=0, atol=1
    )
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


@pytest.mark.parametrize("seed", range(100))
def test_random_garbage(seed):
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

    trip_xyzdt = np.empty((n_points, 5))
    trip_xyzdt[:, 0] = rng.uniform(6.9, 7.1, n_points)
    trip_xyzdt[:, 1] = rng.uniform(39.9, 40.1, n_points)
    trip_xyzdt[:, 2] = z
    trip_xyzdt[:, 3] = rng.uniform(-100, 1100, n_points)
    mean_dist = 1000 / (n_points - 1)
    trip_xyzdt[:, 4] = np.cumsum(
        rng.uniform(-0.1 * mean_dist, 1.1 * mean_dist, n_points)
    )

    # make some distances and times missing
    for i in (3, 4):
        n_missing = rng.integers(0, n_points + 1)
        trip_xyzdt[rng.choice(n_points, n_missing, replace=False), i] = np.nan

    dtrip = DubiousTrajectoryTrip(dtraj, trip_xyzdt)
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    min_spacing = min(
        trip.snapping_params.min_spacing, trip.trajectory.length / (n_points - 1)
    )
    for split_segment in snapped.get_inter_point_ls_lon_lat_in_travel_direction():
        assert (
            WGS84_GEOD.line_length(split_segment[:, 0], split_segment[:, 1])
            >= 0.99 * min_spacing
        )


@pytest.mark.xfail(
    reason="""Data contains a hard case with a trip with some detours, no distance information
and sloppy times (often the same time for different stops) leading to initial distances
ending up in the wrong branch of their detour.
This is one of the few examples that currently still fail to snap with stops close
enough to their original position with the iterative method.
Possible solution: Use different minimum spacings calculated from stop-stop distances
and stop-trajectory distances (currently we only have a fixed minimum spacing of 25 meters).
This could help to get a better initial guess when times are bad.
Or: Fix geOps routing API when skipping stops: Currently all distance information is lost
when stops are skipped during routing. This could be fixed such that only the distances
of the missing stops are missing.
"""
)
def test_complex_trip():
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
        [
            f["geometry"]["coordinates"] + [None, f["properties"]["time"]]
            for f in point_features
        ],
    )
    trip = dtrip.to_trajectory_trip()
    snapped = trip.snap_trip_points()
    assert not snapped.reverse_order
    snapped.raise_invalid()


@pytest.mark.parametrize(
    "stop_dist,expected_trusted,expected_dist",
    (
        (5, True, 5),  # trust and keep
        (35, False, 35),  # don't trust but keep
        (75, False, 0),  # don't trust, don't keep (0 is expected fallback estimate)
    ),
)
def test_snapping_dists_admissible(stop_dist, expected_trusted, expected_dist):
    real_segment_lengths = [5000]
    start = (10, 50)
    dtraj = make_dubious_traj(real_segment_lengths, [0], [0, 5000], start=start)
    trip_xyzdt = np.full((2, 5), np.nan)

    # put first stop 25 meters away from trajectory start
    # in the opposite direction as trajectory runs
    trip_xyzdt[0, :2] = offset(start, 25, 180)
    trip_xyzdt[0, 2] = 0

    # put second stop at end of trajectory (irrelevant for test)
    trip_xyzdt[1, :3] = dtraj.xyz[1]

    # set up external distance along trajectory for stops
    trip_xyzdt[0, 3] = stop_dist

    params = SnappingParams(rtol_trusted=1, atol_trusted=11, rtol_keep=2, atol_keep=15)
    dtrip = DubiousTrajectoryTrip(dtraj, trip_xyzdt)
    trip = dtrip.to_trajectory_trip(snapping_params=params)
    assert trip.dists_trusted[0] == expected_trusted
    assert abs(trip.dists[0] - expected_dist) < 0.1  # 10 cm accuracy
