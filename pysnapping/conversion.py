import sys
import logging
import typing

import numpy as np
from numpy.typing import ArrayLike
from shapely.geometry import LineString
import pyproj
from scipy.interpolate import interp1d

from .snap import (
    Trajectory,
    TrajectoryTrip,
    WGS84Trajectory,
    WGS84TrajectoryTrip,
    WGS84_CRS,
)
from .ordering import fix_sequence_with_missing_values
from .interpolate import fix_repeated_x
from .util import iter_consecutive_groups


logger = logging.getLogger(__name__)


def transform_coords(
    coords: ArrayLike,
    trafo: typing.Callable[
        [np.ndarray, np.ndarray], typing.Tuple[np.ndarray, np.ndarray]
    ],
    out: typing.Optional[np.ndarray] = None,
) -> np.ndarray:
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.shape[-1] != 2:
        raise ValueError("last axis has to be of length 2")
    if out is None:
        out = np.empty_like(coords_arr)
    out[..., 0], out[..., 1] = trafo(coords_arr[..., 0], coords_arr[..., 1])
    return out


def simplify_2d_keep_z(
    coords: ArrayLike, tolerance, fake_nan=sys.float_info.max
) -> np.ndarray:
    """Simplify line string coords using Ramer-Douglas-Peucker algorithm.

    The z-dimension is not considered but it is kept intact for points which
    were not removed. Also deals correctly with NaNs in the z-dimension.
    +-inf is not allowed in the z-dimension.

    `fake_nan` is an arbitrary finite floating point number that should not
    occur in the z-dimension values.
    """
    # TODO (nice to have): this is the only part in the project, where we depend on shapely.
    # Maybe we could implement our own variant of Ramer-Douglas-Peucker algorithm without
    # this ugly hack to get rid of shapely.

    # shapely seems to silently loose the z-dimension in simplify
    # if there are NaNs or infs present :(
    coords = np.array(coords, dtype=float)
    if len(coords.shape) != 2 or coords.shape[1] != 3:
        raise ValueError(f"expected 3d LineString coords, got shape {coords.shape}")
    coords[np.isnan(coords[:, 2]), 2] = fake_nan

    if not np.all(np.isfinite(coords[:, 2])):
        raise ValueError("+-inf not allowed in z dimension")

    simple_coords = np.array(
        LineString(coords).simplify(tolerance=tolerance, preserve_topology=False).coords
    )
    if simple_coords.shape[1] != 3:
        raise RuntimeError("shapely simplify lost the z dimension")

    simple_coords[simple_coords[:, 2] == fake_nan, 2] = np.nan
    return simple_coords


def simplify_trajectory(
    trajectory: Trajectory, tolerance: float, fake_nan: float = sys.float_info.max
) -> Trajectory:
    return Trajectory(
        simplify_2d_keep_z(trajectory.xyd, tolerance, fake_nan), trajectory.crs
    )


def estimate_geodesic_distances(
    trip: TrajectoryTrip,
    min_spacing: float = 25.0,
    max_move_trusted: float = 15.0,
    atol: float = 5.0,
    reverse_order_allowed: bool = True,
    distrust_all_distances: bool = False,
) -> WGS84TrajectoryTrip:
    """Convert a TrajectoryTrip to a WGS84TrajectoryTrip.

    A TrajectoryTrip can contain missing/bogus distance data.
    A WGS84TrajectoryTrip has mandatory, properly ordered geodesic distances
    (yet they can be an estimation depending on the quality of the input data).
    A WGS84TrajectoryTrip is suitable for snapping the points to the trajectory.

    All distance related parameters are in meters.

    `min_spacing` tells how far apart point distances have to be at least.

    `max_move_trusted` tells how far a finite point distance may move from
    its original value in order to still be trusted. Distances might be moved
    to make space for other points or to force them inside the range
    of the trajectory.

    `atol` sets the corresponding attribute on the result.

    Setting `reverse_order_allowed` to `False` will set the corresponding flag on the
    result to `False`. When set to `True` (the default), the flag on the result will
    be determined automatically.
    """
    if len(trip) < 2:
        raise ValueError("at least two trip points are required")
    reverse_order_allowed = reverse_order_allowed and trip.get_reverse_order_allowed()
    wgs84_trip_lat_lon_d = np.empty_like(trip.xydt[:, :3])
    if trip.trajectory.crs != WGS84_CRS:
        trafo = pyproj.Transformer.from_crs(trip.trajectory.crs, WGS84_CRS).transform
        traj_lat_lon = transform_coords(trafo, trip.trajectory.xy)
        transform_coords(trafo, trip.xy, out=wgs84_trip_lat_lon_d[:, :2])
    else:
        traj_lat_lon = trip.trajectory.xy
        wgs84_trip_lat_lon_d[:, :2] = trip.xy
    wgs84_traj = WGS84Trajectory(traj_lat_lon)

    finite_traj_indices = np.nonzero(np.isfinite(trip.trajectory.dists))[0]
    finite_traj_dists = trip.trajectory.dists[finite_traj_indices]

    wgs84_trip_dists = wgs84_trip_lat_lon_d[:, 2]
    if len(finite_traj_dists) >= 1 and np.all(np.diff(finite_traj_dists) >= 0):
        x = finite_traj_dists
        y = wgs84_traj.dists[finite_traj_indices]
        x, y = fix_repeated_x(x, y, axis=0)
        # trip.dists can contain NaNs and +-inf but interp1d only deals with NaN
        x_prime = np.nan_to_num(trip.dists, nan=np.nan, posinf=np.nan, neginf=np.nan)
        if len(x) >= 2:
            wgs84_trip_dists[...] = interp1d(
                x,
                y,
                copy=False,
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )(x_prime)
        else:
            wgs84_trip_dists.fill(np.nan)
            wgs84_trip_dists[x_prime == x[0]] = y[0]
    else:
        wgs84_trip_dists.fill(np.nan)

    # now we have everything together except that the trip distances might still
    # be out of range, too close together, or contain NaNs, so let's fix that
    wgs84_trip_dists[...], trip_dists_trusted = fix_wgs84_trip_distances(
        dists=wgs84_trip_dists,
        times=trip.times,
        d_min=wgs84_traj.d_min,
        d_max=wgs84_traj.d_max,
        min_spacing=min_spacing,
        max_move_trusted=max_move_trusted,
    )

    if distrust_all_distances:
        trip_dists_trusted.fill(False)

    return WGS84TrajectoryTrip(
        trajectory=wgs84_traj,
        lat_lon_d=wgs84_trip_lat_lon_d,
        dists_trusted=trip_dists_trusted,
        min_spacing=min_spacing,
        reverse_order_allowed=reverse_order_allowed,
        atol=atol,
    )


def fix_wgs84_trip_distances(
    dists: np.ndarray,
    times: np.ndarray,
    d_min: float,
    d_max: float,
    min_spacing: float = 25.0,
    # slightly greater than default min_spacing / 2, so making space for points
    # that lie on top of each other is OK
    max_move_trusted: float = 15.0,
    atol: float = 5.0,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Estimate missing trip distances and force correct range and minimum spacing.

    The order and minimum spacing between dists is fixed automatically,
    minimizing the sum of squares of deviations to the input data.
    If dists change too much during this process, they are marked as untrusted.

    `times` is an array containing numbers that give the departure/arrival time
    at the points/stops in an arbitrary but consistent unit with an arbitrary offset
    under the assumption that the vehicle would not wait at the stop (arrival = departure time).
    Or in other words, the cumulative travel times. It can contain NaNs.
    If the finite values are all in increasing order (not necessarily strictly increasing),
    `times` is used to estimate missing point distances where possible.
    The estimated distances are again forced to fall inside the correct range with the correct
    minimum spacing.

    Afterwards, point distances that are still missing are filled equidistantly
    between the sourrounding finite values using `d_min` and `d_max` as a
    fallback at the boundaries.

    All estimated point distances are marked as untrusted.

    Raise `NoSolution` if there are too many points to fit between [d_min, dmax] with
    the given min_spacing.

    Return `(fixed_dists, trusted)` where `trusted` is a boolean mask indicating which
    distance can be trusted (versus was estimated or moved too much).
    """
    if len(dists) < 2:
        raise ValueError("At least two distances are required")
    dists = dists.copy()
    missing = np.isnan(dists)
    available = np.logical_not(missing)

    fixed_dists = fix_sequence_with_missing_values(
        values=dists,
        v_min=d_min,
        v_max=d_max,
        d_min=min_spacing,
        atol=atol / 2,
    )
    not_too_far_away = np.abs(fixed_dists - dists) <= max_move_trusted
    trusted = available & not_too_far_away
    dists = fixed_dists

    # Now the finite point dists are all in order and there is enough space between them
    # and enough space for the missing dists and we have at least two unique known dists.
    # This allows us to use the finite dists for interpolating the missing distances.

    n_available = available.sum()

    # We assume the trip spans the whole trajectory if we don't know enough distances.
    # If we know enough distances, but start/end distances are missing, we prefer extrapolating
    # based on times over this assumption.
    if n_available < 2:
        for index, value in ((0, d_min), (-1, d_max)):
            if missing[index]:
                missing[index] = False
                available[index] = True
                dists[index] = value
                n_available += 1

    # shortcut if no dists are missing
    if n_available == len(dists):
        return (dists, trusted)

    times_available = np.isfinite(times)
    input_indices = np.nonzero(available & times_available)[0]
    output_indices = np.nonzero(missing & times_available)[0]
    if (
        len(output_indices) != 0
        and len(input_indices) >= 2
        # We tolerate zero travel times between stops since this often occurs due to
        # rounding to full minutes.
        # We later fix this by forcing min_spacing on the interpolated point distances.
        # If any travel time is however decreasing, we assume all travel times are garbage,
        # falling back to equidistant interpolation since there is no obvious way
        # to fix negative travel times.
        and np.all(np.diff(times[times_available]) >= 0)
    ):
        x = times[input_indices]
        y = dists[input_indices]
        x, y = fix_repeated_x(x, y, axis=0)
        if len(x) >= 2:  # at least two unique times?
            x_prime = times[output_indices]
            y_prime = interp1d(
                x,
                y,
                copy=False,
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )(x_prime)
            dists[output_indices] = y_prime
            # dists estimated from travel times could be too close together
            # or too close to known dists or out of bounds (if extrapolated)
            _fix_travel_time_dists(
                missing,
                dists,
                d_min,
                d_max,
                min_spacing,
                atol,
            )
            missing[output_indices] = False
            available[output_indices] = True
            n_available += len(output_indices)

    # Now interpolate the remaining missing dists equidistantly (i.e. using indices as x values).
    if n_available != len(dists):
        # In contrast to the time case, we always fix the distances at the boundaries
        # if they are unknown even if we have 2 or more known points.
        # (We think assuming that the trip spans the whole trajectory is a better guess than
        #  assuming that the stops to the left/right of the first/last known stop continue
        #  with the same spacing.)
        for index, value in ((0, d_min), (-1, d_max)):
            if missing[index]:
                missing[index] = False
                available[index] = True
                dists[index] = value
                n_available += 1

        if n_available != len(dists):
            assert n_available >= 2
            x = np.nonzero(available)[0]
            y = dists[available]
            x_prime = np.nonzero(missing)[0]
            y_prime = interp1d(
                x,
                y,
                copy=False,
                bounds_error=True,
                assume_sorted=True,
            )(x_prime)
            dists[missing] = y_prime
            # last change, don't update available, n_available, missing

    return (dists, trusted)


def _fix_travel_time_dists(mask, dists, d_min, d_max, min_spacing, atol):
    n_points = len(dists)
    indices = np.nonzero(mask)[0]
    # every group of consecutive indices is an isolated problem
    for group_indices in iter_consecutive_groups(indices):
        i_left = group_indices[0] - 1
        i_right = group_indices[-1] + 1
        v_min = dists[i_left] + min_spacing if i_left >= 0 else d_min
        v_max = dists[i_right] - min_spacing if i_right < n_points else d_max
        dists[group_indices] = fix_sequence_with_missing_values(
            values=dists[group_indices],
            v_min=v_min,
            v_max=v_max,
            d_min=min_spacing,
            atol=atol / 2,
        )
