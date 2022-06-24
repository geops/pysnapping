import sys
import logging
import typing
from functools import lru_cache

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
    SnappingParams,
    DEFAULT_SNAPPING_PARAMS,
)
from .interpolate import fix_repeated_x
from . import EPSG4326


logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def get_trafo(
    from_crs: pyproj.CRS, always_xy: bool
) -> typing.Callable[[np.ndarray, np.ndarray], typing.Tuple[np.ndarray, np.ndarray]]:
    return pyproj.Transformer.from_crs(
        from_crs,
        EPSG4326,
        always_xy=always_xy,
    ).transform


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
    snapping_params: SnappingParams = DEFAULT_SNAPPING_PARAMS,
) -> WGS84TrajectoryTrip:
    """Convert a TrajectoryTrip to a WGS84TrajectoryTrip.

    A TrajectoryTrip can contain missing/bogus distance data.
    A WGS84TrajectoryTrip has mandatory, properly ordered geodesic distances
    (yet they can be an estimation depending on the quality of the input data).
    A WGS84TrajectoryTrip is suitable for snapping the points to the trajectory.

    All distance related parameters are in meters.

    `snapping_params`: see pysnapping.snap.SnappingParams. Attention: they will be copied
    """
    if len(trip) < 2:
        raise ValueError("at least two trip points are required")
    wgs84_trip_lon_lat_d = np.empty_like(trip.xydt[:, :3])

    # `is` is way fastern than `==` for CRS and by default, `is` holds
    if trip.trajectory is EPSG4326 or trip.trajectory.crs == EPSG4326:
        if trip.trajectory.strict_axis_order:
            # input is lat,lon
            traj_lon_lat = trip.trajectory.xy[:, ::-1]
            wgs84_trip_lon_lat_d[:, :2] = trip.xy[:, ::-1]
        else:
            # input is lon,lat
            traj_lon_lat = trip.trajectory.xy
            wgs84_trip_lon_lat_d[:, :2] = trip.xy
    else:
        trafo = get_trafo(
            trip.trajectory.crs,
            not trip.trajectory.strict_axis_order,
        )
        if trip.trajectory.strict_axis_order:
            # trafo output is lat,lon
            # but we write into views with swapped axis order, so we don't have to copy anything
            traj_lon_lat = np.empty_like(trip.trajectory.xy)
            transform_coords(trip.trajectory.xy, trafo, out=traj_lon_lat[:, ::-1])
            # 1::-1 writes to axes 1 (lat), 0 (lon); axis 2 is untouched (distance)
            transform_coords(trip.xy, trafo, out=wgs84_trip_lon_lat_d[:, 1::-1])
        else:
            # trafo output is lon,lat
            traj_lon_lat = transform_coords(trip.trajectory.xy, trafo)
            transform_coords(trip.xy, trafo, out=wgs84_trip_lon_lat_d[:, :2])

    wgs84_traj = WGS84Trajectory(traj_lon_lat)

    finite_traj_indices = np.nonzero(np.isfinite(trip.trajectory.dists))[0]
    finite_traj_dists = trip.trajectory.dists[finite_traj_indices]

    wgs84_trip_dists = wgs84_trip_lon_lat_d[:, 2]
    if len(finite_traj_dists) >= 1 and np.all(np.diff(finite_traj_dists) >= 0):
        x = finite_traj_dists
        y = wgs84_traj.dists[finite_traj_indices]
        x, y = fix_repeated_x(x, y)
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
    # be out of range, too close together, too far from the point coords or contain NaNs,
    # so let's fix that
    return WGS84TrajectoryTrip.from_invalid_dists_and_times(
        trajectory=wgs84_traj,
        lon_lat_d=wgs84_trip_lon_lat_d,
        times=trip.times,
        snapping_params=snapping_params,
    )
