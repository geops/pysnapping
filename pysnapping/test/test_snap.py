import numpy as np
from numpy.testing import assert_allclose
import pytest
import pyproj
from itertools import product

from pysnapping.snap import DubiousTrajectory, DubiousTrajectoryTrip
from pysnapping.linear_referencing import locate, interpolate


WGS84_GEOD = pyproj.Geod(ellps="WGS84")


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
