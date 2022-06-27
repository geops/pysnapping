import numpy as np
from numpy.testing import assert_allclose
import pytest

from pysnapping import WGS84_GEOD
from pysnapping.snap import DubiousTrajectory, DubiousTrajectoryTrip
from pysnapping.linear_referencing import locate, interpolate


def offset(point, dist, azimuth):
    return np.array(WGS84_GEOD.fwd(*point, az=azimuth, dist=dist)[:2], dtype=float)


def make_dubious_traj(real_segment_lengths, azimuths, dubious_dists, start=(7, 40)):
    points = [start]
    for dist, az in zip(real_segment_lengths, azimuths):
        points.append(offset(points[-1], dist, az))
    return DubiousTrajectory([[p[0], p[1], d] for p, d in zip(points, dubious_dists)])


def segment_lengths_to_dists(segment_lengths, start=0):
    dists = np.full((len(segment_lengths) + 1,), start)
    dists[1:] += np.cumsum(segment_lengths)
    return dists


def test_convert_good_dists():
    real_segment_lengths = [100, 1500, 200, 26]
    dtraj = make_dubious_traj(
        real_segment_lengths, [0, 15, -15, 20], [10, 20, 30, 40, 50]
    )
    dubious_trip_dists = [15, 37.5]
    trip_xydt = [
        tuple(offset(interpolate(dtraj.xy, locate(dtraj.dists, d)), 30, 90)) + (d, None)
        for d in dubious_trip_dists
    ]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xydt=trip_xydt,
    )
    trip = dtrip.to_wgs84_trajectory_trip()

    assert_allclose(trip.trajectory.lon_lat, dtraj.xy)
    assert_allclose(
        trip.trajectory.dists, segment_lengths_to_dists(real_segment_lengths)
    )

    assert_allclose(trip.lon_lat, dtrip.xy)

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

    trip_xydt = [
        (
            tuple(offset(interpolate(dtraj.xy, locate(dtraj.dists, d)), 30, 90))
            if d is not None
            else (7, 40)
        )
        + (d, t)
        for d, t in zip(dists, times)
    ]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xydt=trip_xydt,
    )
    trip = dtrip.to_wgs84_trajectory_trip()
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

    trip_xydt = [(7, 40, None, t) for t in times]
    dtrip = DubiousTrajectoryTrip(
        trajectory=dtraj,
        xydt=trip_xydt,
    )
    trip = dtrip.to_wgs84_trajectory_trip()
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
        segment_lengths = np.array([100])
        n_points = 50
    elif seed == 1:
        # special case: zero length trajectory ("elevator")
        n_segments = 1
        segment_lengths = np.array([0])
        n_points = 3
    else:
        n_segments = rng.integers(1, 1001)
        segment_lengths = rng.uniform(0, 3000, n_segments)
        n_points = rng.integers(2, 51)

    azimuths = rng.uniform(0, 360, n_segments)
    dtraj_dists = np.linspace(0, 1000, n_segments + 1)
    dtraj = make_dubious_traj(segment_lengths, azimuths, dtraj_dists)

    trip_xydt = np.empty((n_points, 4))
    trip_xydt[:, 0] = rng.uniform(6.9, 7.1, n_points)
    trip_xydt[:, 1] = rng.uniform(39.9, 40.1, n_points)
    trip_xydt[:, 2] = rng.uniform(-100, 1100, n_points)
    mean_dist = 1000 / (n_points - 1)
    trip_xydt[:, 3] = np.cumsum(
        rng.uniform(-0.1 * mean_dist, 1.1 * mean_dist, n_points)
    )

    # make some distances and times missing
    for i in (2, 3):
        n_missing = rng.integers(0, n_points + 1)
        trip_xydt[rng.choice(n_points, n_missing, replace=False), i] = np.nan

    dtrip = DubiousTrajectoryTrip(dtraj, trip_xydt)
    trip = dtrip.to_wgs84_trajectory_trip()
    snapped = trip.snap_trip_points()

    min_spacing = min(
        trip.snapping_params.min_spacing, trip.trajectory.length / (n_points - 1)
    )
    for split_segment in snapped.get_inter_point_ls_lon_lat_in_travel_direction():
        assert (
            WGS84_GEOD.line_length(split_segment[:, 0], split_segment[:, 1])
            >= 0.99 * min_spacing
        )
