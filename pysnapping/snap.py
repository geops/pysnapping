import typing
from functools import partial
import logging
from enum import Enum
import json

import pyproj
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from .ordering import order_ok, fix_sequence_with_missing_values
from .linear_referencing import (
    Location,
    substring,
    locate,
    interpolate,
    ProjectionTarget,
)
from .util import (
    iter_consecutive_groups,
    fix_repeated_x,
    get_trafo,
    transform_coords,
    array_chk,
    simplify_2d_keep_z,
)
from . import SnappingError, NoSolution, WGS84_GEOD, EPSG4326


logger = logging.getLogger(__name__)


class SnappingParams(typing.NamedTuple):
    """Parameters controlling snapping of trip points to the trajectory.

    An external location is a point on the trajectory for a trip point.

    If an external location is trusted, snapping will occur exactly to this location.
    If an external location is distrusted but kept, the external location serves as a hint
    where the snapping should occur.
    If an external location is discarded or missing, a new untrusted location will be estimated
    from trip timings with a fallback to equidistant placement between
    surrounding trusted locations and the estimated location serves as a hint where
    the snapping should occur.

    All distances are in meters.

    min_spacing:
    The minimum travel distance along the trajectory between consecutive snapped points.

    atol_spacing:
    Absolute tolerance for checking if distances are admissible (correct min spacing and
    inside trajectory bounds).

    max_move_trusted:
    When fixing minimum spacing/order: how far is the location allowed to move along the trajectory
    to still count as trusted?

    max_move_keep:
    When fixing minimum spacing/order: how far is the location allowed to move along the trajectory
    to still be kept at all?

    rtol_trusted, atol_trusted:
    Let d_min be the minimum distance of the point to the trajectory and d the distance
    of the point to the external location on the trajectory.
    If d <= rtol_trusted * d_min + atol_trusted, the external location is trusted.

    rtol_keep, atol_keep:
    Same as above but when the snapping distance is too large, the location is discarded.

    reverse_order_allowed:
    Whether to also try reversing the trajectory.
    Only applies if no external locations are trusted.

    short_trajectory_fallback:
    If enabled and the trajectory is too short to snap all points
    with min_spacing, points will be placed equidistantly along the whole trajectory.
    """

    min_spacing: float = 25.0
    atol_spacing: float = 5.0
    max_move_trusted: float = 15.0
    max_move_keep: float = 300.0
    rtol_trusted: float = 1.5
    atol_trusted: float = 10.0
    rtol_keep: float = 2.5
    atol_keep: float = 300.0
    reverse_order_allowed: bool = True
    short_trajectory_fallback: bool = True

    def dists_ok(
        self,
        distances: ArrayLike,
        d_min: float,
        d_max: float,
        min_spacing: typing.Optional[float] = None,
        atol: typing.Optional[float] = None,
    ) -> bool:
        if min_spacing is None:
            min_spacing = self.min_spacing
        if atol is None:
            atol = self.atol_spacing
        return order_ok(distances, d_min, d_max, min_spacing, atol)


DEFAULT_SNAPPING_PARAMS = SnappingParams()


class DistanceType(Enum):
    trusted = "trusted"
    projected = "projected"
    iterative = "iterative"


class DubiousTrajectory:
    """A linestring with dubious travel distances for its vertices.

    The distances are to be interpreted as travel distances along
    the linestring in an arbitrary unit with arbitrary offset.

    Distances are allowed to contain NaNs and they are allowed to decrease (thus "dubious").

    The interpretation of the coordinates is given by `crs` (default: epsg:4326).
    By default, traditional GIS order is used for the axis order,
    that is longitude, latitude for geographic CRS and easting, northing for most projected CRS.
    If you want strict axis order as defined by the CRS, set `strict_axis_order` to True.
    """

    __slots__ = ("crs", "xyd", "strict_axis_order")

    crs: pyproj.CRS
    xyd: np.ndarray
    strict_axis_order: bool

    def __init__(
        self,
        xyd: ArrayLike,
        crs: pyproj.CRS = EPSG4326,
        strict_axis_order: bool = False,
        simplify_tolerance: typing.Optional[float] = None,
    ):
        self.crs = crs
        self.strict_axis_order = strict_axis_order
        if simplify_tolerance is not None:
            self.xyd = simplify_2d_keep_z(xyd, simplify_tolerance)
        else:
            self.xyd = array_chk(xyd, ((2, None), 3), dtype=float)
        if not np.all(np.isfinite(self.xyd[:, :2])):
            raise ValueError("coords have to be finite")

    def __len__(self) -> int:
        return len(self.xyd)

    @property
    def xy(self) -> np.ndarray:
        return self.xyd[:, :2]

    @property
    def x(self) -> np.ndarray:
        return self.xyd[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyd[:, 1]

    @property
    def dists(self) -> np.ndarray:
        return self.xyd[:, 2]


class DubiousTrajectoryTrip:
    """A sequence of points along a DubiousTrajectory with dubious distance and time information.

    The interpretation of the coordinates is given by `trajectory.crs`.
    Points do not have to lie on the trajectory.

    The distances are to be interpreted as in `trajectory`.
    Distances are allowed to contain NaNs. They also may fall outside the range
    of the distances of the trajectory and they are allowed to decrease.

    The order of the points has to match the canonical travel direction of the trajectory.
    Exception: If all distances are missing, the points are allowed to be in reverse order.

    The times are numbers that give the departure/arrival time
    at the points in an arbitrary but consistent unit with an arbitrary offset
    under the assumption that there is no waiting time at the points.
    Or in other words, times are the cumulative travel times.
    Times can contain NaNs and times are allowed to decrease.
    """

    __slots__ = ("trajectory", "xydt")

    trajectory: DubiousTrajectory
    xydt: np.ndarray

    def __init__(self, trajectory: DubiousTrajectory, xydt: ArrayLike):
        self.trajectory = trajectory
        self.xydt = array_chk(xydt, (None, 4), dtype=float)
        if not np.all(np.isfinite(self.xydt[:, :2])):
            raise ValueError("coords have to be finite")

    def __len__(self) -> int:
        return len(self.xydt)

    @property
    def dists(self) -> np.ndarray:
        return self.xydt[:, 2]

    @property
    def xy(self) -> np.ndarray:
        return self.xydt[:, :2]

    @property
    def x(self) -> np.ndarray:
        return self.xydt[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xydt[:, 1]

    @property
    def times(self) -> np.ndarray:
        return self.xydt[:, 3]

    def to_wgs84_trajectory_trip(
        self,
        snapping_params: SnappingParams = DEFAULT_SNAPPING_PARAMS,
    ) -> "WGS84TrajectoryTrip":
        """Convert to a WGS84TrajectoryTrip.

        A DubiousTrajectoryTrip can contain missing/bogus distance data.
        A WGS84TrajectoryTrip has mandatory, properly ordered geodesic distances
        (yet they can be an estimation depending on the quality of the input data).
        A WGS84TrajectoryTrip is suitable for snapping the points to the trajectory.

        All distance related parameters are in meters.

        `snapping_params`: see SnappingParams.
        """
        if len(self) < 2:
            raise ValueError("at least two trip points are required")
        wgs84_trip_lon_lat_d = np.empty_like(self.xydt[:, :3])

        # `is` is way fastern than `==` for CRS and by default, `is` holds
        if self.trajectory is EPSG4326 or self.trajectory.crs == EPSG4326:
            if self.trajectory.strict_axis_order:
                # input is lat,lon
                traj_lon_lat = self.trajectory.xy[:, ::-1]
                wgs84_trip_lon_lat_d[:, :2] = self.xy[:, ::-1]
            else:
                # input is lon,lat
                traj_lon_lat = self.trajectory.xy
                wgs84_trip_lon_lat_d[:, :2] = self.xy
        else:
            trafo = get_trafo(
                self.trajectory.crs,
                not self.trajectory.strict_axis_order,
            )
            if self.trajectory.strict_axis_order:
                # trafo output is lat,lon
                # but we write into views with swapped axis order, so we don't have to copy anything
                traj_lon_lat = np.empty_like(self.trajectory.xy)
                transform_coords(self.trajectory.xy, trafo, out=traj_lon_lat[:, ::-1])
                # 1::-1 writes to axes 1 (lat), 0 (lon); axis 2 is untouched (distance)
                transform_coords(self.xy, trafo, out=wgs84_trip_lon_lat_d[:, 1::-1])
            else:
                # trafo output is lon,lat
                traj_lon_lat = transform_coords(self.trajectory.xy, trafo)
                transform_coords(self.xy, trafo, out=wgs84_trip_lon_lat_d[:, :2])

        wgs84_traj = WGS84Trajectory(traj_lon_lat)

        finite_traj_indices = np.nonzero(np.isfinite(self.trajectory.dists))[0]
        finite_traj_dists = self.trajectory.dists[finite_traj_indices]

        wgs84_trip_dists = wgs84_trip_lon_lat_d[:, 2]
        if len(finite_traj_dists) >= 1 and np.all(np.diff(finite_traj_dists) >= 0):
            x = finite_traj_dists
            y = wgs84_traj.dists[finite_traj_indices]
            x, y = fix_repeated_x(x, y)
            # self.dists can contain NaNs and +-inf but interp1d only deals with NaN
            x_prime = np.nan_to_num(
                self.dists, nan=np.nan, posinf=np.nan, neginf=np.nan
            )
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
        return WGS84TrajectoryTrip.from_dubious_dists_and_times(
            trajectory=wgs84_traj,
            lon_lat_d=wgs84_trip_lon_lat_d,
            times=self.times,
            snapping_params=snapping_params,
        )


class WGS84Trajectory:
    """A linestring on the WGS84 ellipsoid with geodesic travel distances.

    Distances are the cumulative geodesic distances along the trajectory in meters.
    Distances do not have to start at 0. An arbitrary offset is allowed. This is useful
    for relating parts of a trajectory to the original trajectory.
    """

    __slots__ = ("lon_lat_d",)

    lon_lat_d: np.ndarray

    def __init__(self, lon_lat: ArrayLike):
        lon_lat = array_chk(
            lon_lat, ((2, None), 2), chk_finite=True, dtype=float, copy=False
        )
        self.lon_lat_d = np.empty((len(lon_lat), 3))
        self.lon_lat[...] = lon_lat

        self.dists[0] = 0
        segment_lengths = WGS84_GEOD.inv(
            lons1=self.lon[:-1],
            lats1=self.lat[:-1],
            lons2=self.lon[1:],
            lats2=self.lat[1:],
        )[2]
        np.cumsum(segment_lengths, out=self.dists[1:])

    def __len__(self) -> int:
        return len(self.lon_lat_d)

    @property
    def d_min(self) -> float:
        return float(self.lon_lat_d[0, 2])

    @property
    def d_max(self) -> float:
        return float(self.lon_lat_d[-1, 2])

    @property
    def lon_lat(self) -> np.ndarray:
        return self.lon_lat_d[:, :2]

    @property
    def lon(self) -> np.ndarray:
        return self.lon_lat_d[:, 0]

    @property
    def lat(self) -> np.ndarray:
        return self.lon_lat_d[:, 1]

    @property
    def dists(self) -> np.ndarray:
        return self.lon_lat_d[:, 2]

    @property
    def length(self) -> float:
        return self.lon_lat_d[-1, 2] - self.lon_lat_d[0, 2]

    @classmethod
    def from_trusted_data(cls, lon_lat_d: ArrayLike, copy: bool = True):
        """Create a WGS84Trajectory from existing coordinates and distances.

        No checks are performed.
        """
        trajectory = cls.__new__(cls)
        trajectory.lon_lat_d = np.array(lon_lat_d, dtype=float, copy=copy)
        return trajectory

    def copy(self):
        return self.__class__.from_trusted_data(self.lon_lat_d)

    def split(
        self, where: typing.Sequence[typing.Union[Location, "WGS84SnappedTripPoint"]]
    ) -> list:
        if len(where) < 2:
            raise ValueError("at least two split positions are required")
        locations = [
            (
                item
                if isinstance(item, Location)
                else self.locate(item.trajectory_distance)
            )
            for item in where
        ]
        return [
            self.__class__.from_trusted_data(
                substring(self.lon_lat_d, locations[i], locations[i + 1])
            )
            for i in range(len(locations) - 1)
        ]

    def locate(self, dist: float, **kwargs) -> Location:
        return locate(self.dists, dist, **kwargs)


class WGS84TrajectoryTrip:
    """A sequence of points along a WGS84Trajectory with mandatory distance information.

    The distances are the geodesic travel distances along the trajectory in meters.
    All distances have to be finite, inside the distance range of the reference trajectory,
    and increasing in steps of at least `snapping_params.min_spacing`.

    `dists_trusted` is a Boolean array indicating whether the corresponding distance is accurate or
    merely a rough hint.

    `snapping_params` controls how snapping will be performed.
    The supplied distances have to pass the `self.dists_ok` test.
    """

    __slots__ = (
        "trajectory",
        "lon_lat_d",
        "dists_trusted",
        "snapping_params",
    )

    trajectory: WGS84Trajectory
    lon_lat_d: np.ndarray
    dists_trusted: np.ndarray
    snapping_params: SnappingParams

    def __init__(
        self,
        trajectory: WGS84Trajectory,
        lon_lat_d: ArrayLike,
        dists_trusted: ArrayLike,
        snapping_params: SnappingParams = DEFAULT_SNAPPING_PARAMS,
    ):
        self.trajectory = trajectory
        self.lon_lat_d = array_chk(lon_lat_d, (None, 3), chk_finite=True, dtype=float)
        self.dists_trusted = array_chk(
            dists_trusted, (len(self.lon_lat_d),), dtype=bool
        )
        self.snapping_params = snapping_params

        if snapping_params.short_trajectory_fallback and self.trajectory_is_too_short:
            if not np.all(self.dists_trusted):
                raise ValueError(
                    "all dists need to be trusted for trajectories that are too short"
                )
            if not self.dists_ok(self.dists, min_spacing=0.0, atol=1e-3):
                raise ValueError("bad distances")
        elif not self.dists_ok(self.dists):
            raise ValueError("bad distances")

    @classmethod
    def from_dubious_dists_and_times(
        cls,
        trajectory: WGS84Trajectory,
        lon_lat_d: ArrayLike,
        times: ArrayLike,
        snapping_params: SnappingParams = DEFAULT_SNAPPING_PARAMS,
    ):
        """Initialize from dubious distances and times.

        Estimate missing trip distances and force correct range and minimum spacing.

        If there are too many points to fit with the given min_spacing and the fallback is enabled,
        points are placed equidistantly and are all marked as trusted.
        If the fallback is disabled, NoSolution is raised.

        If the trajectory is long enough:

        At first, available dists are marked as untrusted or discarded if the snapped point
        would be too far from the trip point according to `snapping_params`.

        Then the order and minimum spacing between dists is fixed automatically,
        minimizing the sum of squares of deviations to the input data.
        If dists change too much during this process, they are marked as untrusted or discarded.

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
        """
        self = cls.__new__(cls)
        self.trajectory = trajectory
        self.lon_lat_d = array_chk(lon_lat_d, (None, 3), dtype=float)
        if not np.all(np.isfinite(self.lon_lat)):
            raise ValueError("coords have to be finite")
        missing = np.isnan(self.dists)
        available = np.logical_not(missing)
        self.dists_trusted = available.copy()
        self.snapping_params = snapping_params
        times = array_chk(times, (len(self),), dtype=float, copy=False)

        self._fix_distances(missing, available, times)

        # is our implementation working?
        assert np.all(np.isfinite(self.dists))

        if snapping_params.short_trajectory_fallback and self.trajectory_is_too_short:
            assert self.dists_ok(self.dists, min_spacing=0.0, atol=1e-3)
        else:
            assert self.dists_ok(self.dists)

        return self

    def __len__(self) -> int:
        return len(self.lon_lat_d)

    @property
    def lon_lat(self) -> np.ndarray:
        return self.lon_lat_d[:, :2]

    @property
    def lon(self) -> np.ndarray:
        return self.lon_lat_d[:, 0]

    @property
    def lat(self) -> np.ndarray:
        return self.lon_lat_d[:, 1]

    @property
    def dists(self) -> np.ndarray:
        return self.lon_lat_d[:, 2]

    @property
    def trajectory_is_too_short(self) -> bool:
        required_length = max(0, len(self) - 1) * self.snapping_params.min_spacing
        # use >= to trigger fallback also when min_spacing is 0 and length is 0
        return required_length >= self.trajectory.length

    def snap_trip_points(
        self,
        convergence_accuracy: float = 1.0,
        n_iter_max: int = 1000,
    ) -> "WGS84SnappedTripPoints":
        """Snap trip points to the trajectory.

        Points with trusted distances end up exactly at those distances.
        Points with untrusted distances are placed acoording to untrusted distances
        and then iteratively optimized such that they get closer to
        the unsnapped point coords but stay on the trajectory and
        keep their order and minimum spacing intact.

        All distances are in meters.

        convergence_accuracy -- stop iterative solution when distances do not move more than this
        """

        snapped_points = np.empty_like(self.dists, dtype=object)
        trusted_indices = np.nonzero(self.dists_trusted)[0]
        params = self.snapping_params
        reverse_order_allowed = (
            len(trusted_indices) == 0 and params.reverse_order_allowed
        )
        is_reversed = False

        # trusted distances
        for i in trusted_indices:
            snapped_points[i] = WGS84SnappedTripPoint(
                self, i, self.dists[i], DistanceType.trusted
            )

        # every group of consecutive points with untrusted distances is an isolated problem
        untrusted_indices = np.nonzero(np.logical_not(self.dists_trusted))[0]
        for group_indices in iter_consecutive_groups(untrusted_indices):
            # if is_reversed is True, the loop has exactly one pass, so we can directly assign
            snapped_points[group_indices], is_reversed = self._snap_untrusted(
                indices=group_indices,
                reverse_order_allowed=reverse_order_allowed,
                convergence_accuracy=convergence_accuracy,
                n_iter_max=n_iter_max,
            )

        return WGS84SnappedTripPoints(snapped_points.tolist(), is_reversed)

    def dists_ok(
        self,
        dists_or_snapped_points: typing.Iterable[
            typing.Union["WGS84SnappedTripPoint", float]
        ],
        d_min: typing.Optional[float] = None,
        d_max: typing.Optional[float] = None,
        min_spacing: typing.Optional[float] = None,
        atol: typing.Optional[float] = None,
    ) -> bool:
        if d_min is None:
            d_min = self.trajectory.d_min
        if d_max is None:
            d_max = self.trajectory.d_max
        return self.snapping_params.dists_ok(
            [
                item.trajectory_distance
                if isinstance(item, WGS84SnappedTripPoint)
                else item
                for item in dists_or_snapped_points
            ],
            d_min=d_min,
            d_max=d_max,
            min_spacing=min_spacing,
            atol=atol,
        )

    def _fix_distances(
        self,
        missing: np.ndarray,
        available: np.ndarray,
        times: np.ndarray,
    ) -> None:
        params = self.snapping_params

        if len(self) < 2:
            raise ValueError("At least two trip points are required")

        # special case: trajectory is too short for snapping all points with min_spacing
        if self.trajectory_is_too_short:
            if params.short_trajectory_fallback:
                # Put everything equidistantly along the whole trajectory.
                # We ignore the possibility that the trajectory might be reversed
                # until this case really pops up.
                logger.info(
                    "falling back to equidistant placement for trajectory that is too short"
                )
                self.dists_trusted.fill(True)
                self.dists[...] = np.linspace(
                    self.trajectory.d_min, self.trajectory.d_max, len(self)
                )
                return
            else:
                raise NoSolution("trajectory is too short, but fallback is disabled")

        # check distances by comparing snapping distance to projection distance
        # and untrust / discard
        available_indices = np.nonzero(available)[0]
        target = ProjectionTarget(self.trajectory.lon_lat)
        for i in available_indices:
            snapped_point = WGS84SnappedTripPoint(
                self, i, self.dists[i], DistanceType.trusted
            )
            snapping_distance = snapped_point.get_geodesic_snapping_distance()
            projection_distance = snapped_point.get_geodesic_projection_distance(target)
            if not snapped_point.snapping_distance_valid(
                rtol=params.rtol_keep,
                atol=params.atol_keep,
                snapping_distance=snapping_distance,
                projection_distance=projection_distance,
            ):
                logger.debug("discarding distance %d: too far away from trip point", i)
                self.dists_trusted[i] = False
                missing[i] = True
                available[i] = False
                self.dists[i] = np.nan
            elif not snapped_point.snapping_distance_valid(
                rtol=params.rtol_trusted,
                atol=params.atol_trusted,
                snapping_distance=snapping_distance,
                projection_distance=projection_distance,
            ):
                logger.debug("distrusting distance %d: too far away from trip point", i)
                self.dists_trusted[i] = False
        del available_indices

        # fix range and minimum spacing and discard/distrust if dists moved too far
        fixed_dists = fix_sequence_with_missing_values(
            values=self.dists,
            v_min=self.trajectory.d_min,
            v_max=self.trajectory.d_max,
            d_min=params.min_spacing,
            atol=params.atol_spacing,
        )
        keep = np.abs(fixed_dists - self.dists) <= params.max_move_keep
        discard = np.logical_not(keep)
        self.dists_trusted &= keep
        missing |= discard
        available &= keep
        fixed_dists[discard] = np.nan
        still_trusted = np.abs(fixed_dists - self.dists) <= params.max_move_trusted
        self.dists_trusted &= still_trusted
        self.dists[...] = fixed_dists

        # Now the finite point dists are all in order and in the correct interval and there is
        # enough space between them and enough space for the missing dists
        # and we have at least two unique known dists (including fallback to trajectory boundaries).
        # This allows us to use the finite dists and times for interpolating the missing distances.

        n_available = available.sum()

        # We assume the trip spans the whole trajectory if we don't know enough distances.
        # If we know enough distances, but start/end distances are missing, we prefer extrapolating
        # based on times over this assumption.
        if n_available < 2:
            for index, value in (
                (0, self.trajectory.d_min),
                (-1, self.trajectory.d_max),
            ):
                if missing[index]:
                    missing[index] = False
                    available[index] = True
                    self.dists[index] = value
                    n_available += 1

        # shortcut if no dists are missing
        if n_available == len(self):
            return

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
            y = self.dists[input_indices]
            x, y = fix_repeated_x(x, y)
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
                self.dists[output_indices] = y_prime
                # dists estimated from travel times could be too close together
                # or too close to known dists or out of bounds (if extrapolated)
                self._fix_travel_time_dists(missing)
                missing[output_indices] = False
                available[output_indices] = True
                n_available += len(output_indices)

        # shortcut if all distances could be interpolated by time
        if n_available == len(self):
            return

        # Now interpolate the remaining missing dists equidistantly
        # (i.e. using indices as x values).
        # In contrast to the time case, we always fix the distances at the boundaries
        # if they are unknown even if we have 2 or more known points.
        # (We think assuming that the trip spans the whole trajectory is a better guess than
        #  assuming that the stops to the left/right of the first/last known stop continue
        #  with the same spacing.)
        for index, value in ((0, self.trajectory.d_min), (-1, self.trajectory.d_max)):
            if missing[index]:
                missing[index] = False
                available[index] = True
                self.dists[index] = value
                n_available += 1

        # shortcut if fixing the boundaries fixed everything
        if n_available == len(self):
            return

        assert n_available >= 2
        x = np.nonzero(available)[0]
        y = self.dists[available]
        x_prime = np.nonzero(missing)[0]
        y_prime = interp1d(
            x,
            y,
            copy=False,
            bounds_error=True,
            assume_sorted=True,
        )(x_prime)
        self.dists[missing] = y_prime
        # last change, don't update available, n_available, missing

    def _fix_travel_time_dists(self, mask: np.ndarray) -> None:
        min_spacing = self.snapping_params.min_spacing
        atol = self.snapping_params.atol_spacing
        d_min = self.trajectory.d_min
        d_max = self.trajectory.d_max
        n_points = len(self)
        indices = np.nonzero(mask)[0]
        # every group of consecutive indices is an isolated problem
        for group_indices in iter_consecutive_groups(indices):
            i_left = group_indices[0] - 1
            i_right = group_indices[-1] + 1
            self.dists[group_indices] = fix_sequence_with_missing_values(
                values=self.dists[group_indices],
                v_min=self.dists[i_left] + min_spacing if i_left >= 0 else d_min,
                v_max=self.dists[i_right] - min_spacing
                if i_right < n_points
                else d_max,
                d_min=min_spacing,
                atol=atol,
            )

    def _snap_untrusted(
        self,
        indices: typing.List[int],
        reverse_order_allowed: bool,
        convergence_accuracy: float,
        n_iter_max: int,
    ) -> typing.Tuple[typing.List["WGS84SnappedTripPoint"], bool]:
        # calculate free space where we can snap to without getting too close
        # to points with trusted dists
        params = self.snapping_params
        left = indices[0] - 1
        right = indices[-1] + 1
        if left < 0:
            d_min = self.trajectory.d_min
        else:
            d_min = self.dists[left] + params.min_spacing
        if right >= len(self):
            d_max = self.trajectory.d_max
        else:
            d_max = self.dists[right] - params.min_spacing
        n_points = len(indices)
        available_length = d_max - d_min
        required_length = (n_points - 1) * params.min_spacing
        if required_length > available_length:
            raise SnappingError("not enough space")

        # the linestring we can snap to without violating d_min / d_max
        target_lon_lat_d = substring(
            self.trajectory.lon_lat_d,
            self.trajectory.locate(d_min),
            self.trajectory.locate(d_max),
        )

        # first try just projecting; if this works, we're done
        target = ProjectionTarget(target_lon_lat_d[:, :2])
        snapped_points = [
            WGS84SnappedTripPoint(
                self,
                i,
                float(
                    interpolate(
                        target_lon_lat_d[:, 2],
                        target.project(self.lon_lat[i]).location,
                    )
                ),
                DistanceType.projected,
            )
            for i in indices
        ]

        d_ok = partial(self.dists_ok, d_min=d_min, d_max=d_max)
        if d_ok(snapped_points):
            logger.debug("projecting in forward direction succesful")
            is_reversed = False
        elif reverse_order_allowed and d_ok(reversed(snapped_points)):
            logger.debug("projecting in reverse direction succesful")
            is_reversed = True
        else:
            logger.debug("projected solution not admissible")
            snapped_points = self._snap_untrusted_iteratively(
                indices=indices,
                d_min=d_min,
                d_max=d_max,
                reverse=False,
                convergence_accuracy=convergence_accuracy,
                n_iter_max=n_iter_max,
            )
            is_reversed = False
            if reverse_order_allowed:
                # pick forward/backward solution with better sum of squared snapping distances
                residuum = sum(
                    p.get_geodesic_snapping_distance() ** 2 for p in snapped_points
                )
                alt_snapped_points = self._snap_untrusted_iteratively(
                    indices=indices,
                    d_min=d_min,
                    d_max=d_max,
                    reverse=True,
                    convergence_accuracy=convergence_accuracy,
                    n_iter_max=n_iter_max,
                )
                alt_residuum = sum(
                    p.get_geodesic_snapping_distance() ** 2 for p in alt_snapped_points
                )
                if alt_residuum < residuum:
                    # we reverse later after dists_ok check
                    is_reversed = True
                    snapped_points = alt_snapped_points

            assert d_ok(
                snapped_points
            ), "bad distances from _snap_untrusted_iteratively"
            if is_reversed:
                snapped_points = snapped_points[::-1]

        return snapped_points, is_reversed

    def _snap_untrusted_iteratively(
        self,
        indices: typing.List[int],
        d_min: float,
        d_max: float,
        reverse: bool,
        convergence_accuracy: float,
        n_iter_max: int,
    ) -> typing.List["WGS84SnappedTripPoint"]:
        """Snap points with untrusted distances iteratively.

        Start with an admissible initial guess for the snapped points.
        Then calculate a region around each snapped point where the point
        is allowed to be such that all points can be placed anywhere inside
        their region without coming too close to any of the other regions or the
        boundaries. Place each snapped point closest to its source point inside its region.
        Then recalculate regions and repeat until converged.

        x: points on trajectory
        |: region boundaries
        s: start
        e: end
        pi: source point i

                  p1                      p2

                   d_min        <-min_spacing->             d_max
                     v                                        v
        s------------|-----x----|-------------|----x----------|----------e

          forbidden    region 1    forbidden       region 2     forbidden

        In the above example, in the next step, both will move to the left,
        then regions are recalculated and the right part can move to the left
        a little further.

        If `reverse` is set to `True`, the order of the points is reversed
        and the initial distances are reflected at `(d_min + d_max) / 2` and reversed.
        """
        params = self.snapping_params
        if reverse:
            indices = indices[::-1]
            point_dists = self.dists[indices]
            point_dists = (d_min + d_max) - point_dists
        else:
            point_dists = self.dists[indices]

        if not self.dists_ok(point_dists, d_min=d_min, d_max=d_max):
            raise ValueError("bad initial untrusted point distances")

        region_boundaries = np.empty(2 * len(indices))
        region_boundaries[0] = d_min
        region_boundaries[-1] = d_max
        n_iter = 0
        # maximum point_dists change compared to last iteration
        delta = float("inf")

        # close to convergence, we move half the last delta in the next step, so if we would
        # run forever, we would still move (delta * sum (1/2 ** n), n=1 to infinity) = delta,
        # thus in all following steps combined, we move at most delta
        # so `convergence_accuracy / 1.1` should be fine to detect convergence with desired accuracy
        while delta > convergence_accuracy / 1.1:
            n_iter += 1
            if n_iter > n_iter_max:
                raise SnappingError("not converged")
            mid_dists = 0.5 * (point_dists[:-1] + point_dists[1:])
            region_boundaries[1:-2:2] = mid_dists - 0.5 * params.min_spacing
            region_boundaries[2:-1:2] = mid_dists + 0.5 * params.min_spacing

            logger.debug(
                "n_iter=%d: delta=%.2e point_dists=%s region_boundaries=%s",
                n_iter,
                delta,
                point_dists,
                region_boundaries,
            )
            # TODO (nice to have/performance):
            # make only one target that supports projecting to regions
            # without actually extracting the substrings
            regions = [
                substring(
                    self.trajectory.lon_lat_d,
                    self.trajectory.locate(region_boundaries[i]),
                    self.trajectory.locate(region_boundaries[i + 1]),
                )
                for i in range(0, len(region_boundaries) - 1, 2)
            ]
            points_projected_to_regions = [
                ProjectionTarget(region[:, :2]).project(self.lon_lat[i])
                for i, region in zip(indices, regions)
            ]
            new_point_dists = np.array(
                [
                    interpolate(region[:, 2], p.location)
                    for p, region in zip(points_projected_to_regions, regions)
                ]
            )
            delta = np.abs(new_point_dists - point_dists).max()
            point_dists = new_point_dists

        logger.debug("converged in %d iterations", n_iter)

        return [
            WGS84SnappedTripPoint(self, i, d, DistanceType.iterative)
            for i, d in zip(indices, point_dists)
        ]


class WGS84SnappedTripPoint:
    __slots__ = (
        "trip",
        "index",
        "trajectory_distance",
        "method",
    )

    trip: WGS84TrajectoryTrip
    index: int
    trajectory_distance: float
    method: DistanceType

    def __init__(
        self,
        trip: WGS84TrajectoryTrip,
        index: int,
        trajectory_distance: float,
        method: DistanceType,
    ):
        self.trip = trip
        self.index = index
        self.trajectory_distance = trajectory_distance
        self.method = method

    def get_trajectory_location(self) -> Location:
        return locate(self.trip.trajectory.dists, self.trajectory_distance)

    def get_lon_lat(self) -> np.ndarray:
        return interpolate(self.trip.trajectory.lon_lat, self.get_trajectory_location())

    def get_geodesic_snapping_distance(self) -> float:
        """Get the geodesic distance between source and snapped point in meters"""
        source_lon_lat = self.trip.lon_lat[self.index]
        snapped_lon_lat = self.get_lon_lat()
        return float(
            WGS84_GEOD.inv(
                lons1=source_lon_lat[0],
                lats1=source_lon_lat[1],
                lons2=snapped_lon_lat[0],
                lats2=snapped_lon_lat[1],
            )[2]
        )

    def get_geodesic_projection_distance(
        self, target: typing.Optional[ProjectionTarget] = None
    ) -> float:
        if target is None:
            target = ProjectionTarget(self.trip.trajectory.lon_lat)
        source_lon_lat = self.trip.lon_lat[self.index]
        projected_lon_lat = target.project(source_lon_lat).coords
        return WGS84_GEOD.inv(
            lons1=source_lon_lat[0],
            lats1=source_lon_lat[1],
            lons2=projected_lon_lat[0],
            lats2=projected_lon_lat[1],
        )[2]

    def snapping_distance_valid(
        self,
        rtol: typing.Optional[float] = None,
        atol: typing.Optional[float] = None,
        target: typing.Optional[ProjectionTarget] = None,
        snapping_distance: typing.Optional[float] = None,
        projection_distance: typing.Optional[float] = None,
    ) -> bool:
        """Determine whether the geodesic snapping distance is valid.

        The geodesic distance between the source and snapped point is compared to the
        (approximate) shortest geodesic distance between the source point and the trajectory.
        The geodesic distance to the trajectory is approximated by the geodesic distance between
        the source point and its eucledian projection onto the trajectory.

        rtol/atol -- See SnappingParams. Default is to use *tol_keep from trip.
        target -- optional pre-computed projection target (for performance optimizations)
        """
        if rtol is None:
            rtol = self.trip.snapping_params.rtol_keep
        if atol is None:
            atol = self.trip.snapping_params.atol_keep
        if snapping_distance is None:
            snapping_distance = self.get_geodesic_snapping_distance()
        if projection_distance is None:
            projection_distance = self.get_geodesic_projection_distance(target)
        diff = abs(snapping_distance - projection_distance)
        return diff <= rtol * projection_distance + atol


class WGS84SnappedTripPoints:
    __slots__ = ("snapped_points", "dists_descending")

    snapped_points: typing.List[WGS84SnappedTripPoint]
    dists_descending: bool

    def __init__(
        self,
        snapped_points: typing.Iterable[WGS84SnappedTripPoint],
        dists_descending: bool,
    ):
        self.snapped_points = list(snapped_points)
        self.dists_descending = dists_descending

    def snapping_distances_valid(
        self, rtol: typing.Optional[float] = None, atol: typing.Optional[float] = None
    ) -> bool:
        """Determine whether all geodesic snapping distances are valid.

        See WGS84SnappedTripPoint.snapping_distance_valid for meaning of parameters.
        """
        if self.snapped_points:
            target = ProjectionTarget(self.snapped_points[0].trip.trajectory.lon_lat)
            return all(
                p.snapping_distance_valid(rtol=rtol, atol=atol, target=target)
                for p in self.snapped_points
            )
        else:
            return True

    def raise_invalid(
        self,
        rtol: typing.Optional[float] = None,
        atol: typing.Optional[float] = None,
        debug_geojson_path: typing.Optional[str] = None,
    ):
        """Check snapping_distances_valid and raise SnappingError if not valid.

        Includes debug GeoJSON in the error message.
        """
        if not self.snapping_distances_valid(rtol=rtol, atol=atol):
            debug = json.dumps(self.to_geojson(), indent=2)
            if debug_geojson_path:
                logger.info("writing debug GeoJSON to %r", debug_geojson_path)
                with open(debug_geojson_path, "w") as handle:
                    handle.write(debug)
            raise SnappingError(
                f"snapped stop too far away from original stop:\n{debug}\n"
            )

    def get_inter_point_trajectories(self) -> typing.List[WGS84Trajectory]:
        """Split the trajectory at the snapped points.

        Return sub-trajectories between consecutive pairs of trip points
        in the same order as in the trip.

        Attention!
        Sub-trajectories are always oriented in the same direction
        as the original trajectory. So if `self.dists_descending` is `True`,
        all sub-trajectories will run from destination point to source point.
        If you only need the coordinates (and not the distances) of the trajectories but oriented
        in travel direction, use `get_inter_point_ls_lon_lat_in_travel_direction` instead.
        """
        if len(self.snapped_points) < 2:
            raise ValueError("at least two snapped points are required")

        trajectory = self.snapped_points[0].trip.trajectory

        if self.dists_descending:
            return trajectory.split(self.snapped_points[::-1])[::-1]
        else:
            return trajectory.split(self.snapped_points)

    def get_inter_point_ls_lon_lat_in_travel_direction(
        self,
    ) -> typing.List[np.ndarray]:
        """Split the trajectory at the snapped points.

        Return linestring lon,lat coords between consecutive pairs of trip points
        in the same order as in the trip.
        All linestrings are oriented in travel direction of the trip.
        """
        trajectories = self.get_inter_point_trajectories()

        # t.lon_lat is a view into t.lon_lat_d, so we copy to free distance data
        if self.dists_descending:
            return [t.lon_lat[::-1].copy() for t in trajectories]
        else:
            return [t.lon_lat.copy() for t in trajectories]

    def to_geojson(self) -> typing.Dict[str, typing.Any]:
        trajectories = self.get_inter_point_trajectories()

        segments = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": t.lon_lat.tolist(),
                },
                "properties": {
                    "what": "partial trajectory",
                    "from_index": int(self.snapped_points[i].index),
                    "to_index": int(self.snapped_points[i + 1].index),
                    "from_distance": float(t.dists[0]),
                    "to_distance": float(t.dists[-1]),
                    "length": float(t.dists[-1] - t.dists[0]),
                },
            }
            for i, t in enumerate(trajectories)
        ]

        snapped_points: typing.List[dict] = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": p.get_lon_lat().tolist(),
                },
                "properties": {
                    "what": "snapped point",
                    "index": int(p.index),
                    "trajectory_distance": float(p.trajectory_distance),
                    "snapping_distance": float(p.get_geodesic_snapping_distance()),
                    "method": p.method.value,
                },
            }
            for p in self.snapped_points
        ]

        trip_points: typing.List[dict] = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": p.trip.lon_lat[p.index].tolist(),
                },
                "properties": {
                    "what": "trip point",
                    "index": int(p.index),
                    "trajectory_distance": float(p.trip.dists[p.index]),
                    "distance_trusted": bool(p.trip.dists_trusted[p.index]),
                },
            }
            for p in self.snapped_points
        ]

        snapping_arrows = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        tp["geometry"]["coordinates"],
                        sp["geometry"]["coordinates"],
                    ],
                },
                "properties": {
                    "what": "snapping arrows",
                    "index": tp["properties"]["index"],
                    "snapping_distance": sp["properties"]["snapping_distance"],
                },
            }
            for tp, sp in zip(trip_points, snapped_points)
        ]

        features = segments
        features.extend(snapped_points)
        features.extend(trip_points)
        features.extend(snapping_arrows)

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {"trajectory_reversed": bool(self.dists_descending)},
        }
