import typing
from functools import partial
import logging
import json

import pyproj
import numpy as np
from numpy.typing import ArrayLike

from .ordering import order_ok, fix_sequence_with_missing_values
from .linear_referencing import (
    Locations,
    substrings,
    locate,
    interpolate,
    ProjectionTarget,
    ProjectedPoints,
    LineFractions,
    resample,
)
from .util import (
    iter_consecutive_groups,
    get_trafo,
    transform_coords,
    array_chk,
    simplify_2d_keep_rest,
)
from . import (
    SnappingError,
    NoSolution,
    ExtrapolationError,
    EPSG4326,
    EPSG4978,
    SnappingMethod,
)


logger = logging.getLogger(__name__)


class XYZDMixin:
    __slots__ = ("xyzd",)

    xyzd: np.ndarray

    def __len__(self) -> int:
        return len(self.xyzd)

    @property
    def xyz(self) -> np.ndarray:
        return self.xyzd[:, :3]

    @property
    def xy(self) -> np.ndarray:
        return self.xyzd[:, :2]

    @property
    def dists(self) -> np.ndarray:
        return self.xyzd[:, 3]


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

    def snapping_dists_admissible(
        self,
        snapping_dists: ArrayLike,
        shortest_dists: ArrayLike,
        rtol: float,
        atol: float,
    ) -> np.ndarray:
        snapping_dists = np.asarray(snapping_dists, dtype=float)
        shortest_dists = np.asarray(shortest_dists, dtype=float)
        return snapping_dists <= rtol * shortest_dists + atol

    def trust_snapping_dists(
        self, snapping_dists: ArrayLike, shortest_dists: ArrayLike
    ) -> np.ndarray:
        return self.snapping_dists_admissible(
            snapping_dists,
            shortest_dists,
            rtol=self.rtol_trusted,
            atol=self.atol_trusted,
        )

    def keep_snapping_dists(
        self, snapping_dists: ArrayLike, shortest_dists: ArrayLike
    ) -> np.ndarray:
        return self.snapping_dists_admissible(
            snapping_dists, shortest_dists, rtol=self.rtol_keep, atol=self.atol_keep
        )


DEFAULT_SNAPPING_PARAMS = SnappingParams()


class DubiousTrajectory(XYZDMixin):
    """A linestring with dubious travel distances for its vertices.

    The distances are to be interpreted as travel distances along
    the linestring in an arbitrary unit with arbitrary offset.

    Distances are allowed to contain NaNs and they are allowed to decrease (thus "dubious").

    The interpretation of the coordinates is given by `crs` (default: epsg:4326).
    By default, traditional GIS order is used for the axis order,
    that is longitude, latitude for geographic CRS and easting, northing for most projected CRS.
    If you want strict axis order as defined by the CRS, set `strict_axis_order` to True.
    """

    __slots__ = ("crs", "strict_axis_order")

    crs: pyproj.CRS
    strict_axis_order: bool

    def __init__(
        self,
        xyzd: ArrayLike,
        crs: pyproj.CRS = EPSG4326,
        strict_axis_order: bool = False,
        simplify_tolerance: typing.Optional[float] = None,
    ):
        """Initialize DubiousTrajectory.

        Attention: So far, simplify only considers x and y values!
        """
        self.crs = crs
        self.strict_axis_order = strict_axis_order
        if simplify_tolerance is not None:
            self.xyzd = simplify_2d_keep_rest(xyzd, simplify_tolerance)
        else:
            self.xyzd = array_chk(xyzd, ((2, None), 4), dtype=float)
        if not np.all(np.isfinite(self.xyz)):
            raise ValueError("coords have to be finite")

    @classmethod
    def from_xyd_and_z(
        cls, xyd: ArrayLike, z: typing.Union[float, ArrayLike] = 0.0, **kwargs
    ):
        xyd_arr = np.asarray(xyd, dtype=float)
        xyzd = np.empty((xyd_arr.shape[0], 4))
        xyzd[:, :2] = xyd_arr[:, :2]
        xyzd[:, 2] = z
        xyzd[:, 3] = xyd_arr[:, 2]
        return cls(xyzd, **kwargs)


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

    __slots__ = ("trajectory", "xyzdt")

    trajectory: DubiousTrajectory
    xyzdt: np.ndarray

    def __init__(self, trajectory: DubiousTrajectory, xyzdt: ArrayLike):
        self.trajectory = trajectory
        self.xyzdt = array_chk(xyzdt, (None, 5), dtype=float)
        if not np.all(np.isfinite(self.xyz)):
            raise ValueError("coords have to be finite")

    def __len__(self) -> int:
        return len(self.xyzdt)

    @property
    def xyz(self) -> np.ndarray:
        return self.xyzdt[:, :3]

    @property
    def xyzd(self) -> np.ndarray:
        return self.xyzdt[:, :4]

    @property
    def dists(self) -> np.ndarray:
        return self.xyzdt[:, 3]

    @property
    def times(self) -> np.ndarray:
        return self.xyzdt[:, 4]

    def to_trajectory_trip(
        self,
        snapping_params: SnappingParams = DEFAULT_SNAPPING_PARAMS,
    ) -> "TrajectoryTrip":
        """Convert to a TrajectoryTrip.

        A DubiousTrajectoryTrip can contain missing/bogus distance data.
        A TrajectoryTrip has mandatory, properly ordered metric distances
        (yet they can be an estimation depending on the quality of the input data).
        A TrajectoryTrip is suitable for snapping the points to the trajectory.

        All distance related parameters are in meters.

        `snapping_params`: see SnappingParams.
        """
        if len(self) < 2:
            raise ValueError("at least two trip points are required")
        trip_xyzd = np.empty_like(self.xyzd)

        # axis order is always the same for EPSG4978 (strict/traditional order)
        if self.trajectory.crs is EPSG4978 or self.trajectory.crs == EPSG4978:
            traj_xyz = self.trajectory.xyz
            trip_xyzd[:, :3] = self.xyz
        else:
            trafo = get_trafo(
                from_crs=self.trajectory.crs,
                strict_axis_order=self.trajectory.strict_axis_order,
            )
            traj_xyz = transform_coords(self.trajectory.xyz, trafo)
            transform_coords(self.xyz, trafo, out=trip_xyzd[:, :3])

        traj = Trajectory(traj_xyz)

        finite_traj_indices = np.nonzero(np.isfinite(self.trajectory.dists))[0]
        finite_traj_dists = self.trajectory.dists[finite_traj_indices]

        trip_dists = trip_xyzd[:, 3]
        resampled = False
        if len(finite_traj_dists) >= 2 and np.all(np.diff(finite_traj_dists) >= 0):
            try:
                trip_dists[...] = resample(
                    x=finite_traj_dists,
                    y=traj.dists[finite_traj_indices],
                    x_prime=self.dists,
                    extrapolate=True,
                )
            except ExtrapolationError:
                pass
            else:
                resampled = True
        if not resampled:
            trip_dists.fill(np.nan)

        # now we have everything together except that the trip distances might still
        # be out of range, too close together, too far from the point coords or contain NaNs,
        # so let's fix that
        return TrajectoryTrip.from_dubious_dists_and_times(
            trajectory=traj,
            xyzd=trip_xyzd,
            times=self.times,
            snapping_params=snapping_params,
        )


class Trajectory(XYZDMixin):
    """A 3d linestring in epsg:4978 with metric travel distances.

    Distances are the cumulative cartesian distances along the trajectory in meters.
    As long as segments are not too long (say a few 100 km per segment), the distances
    are a very good approximation to WGS84 geodesic distances (including altitude changes).
    Distances do not have to start at 0. An arbitrary offset is allowed. This is useful
    for relating parts of a trajectory to the original trajectory.

    Whenever you make changes to `xyzd`, you have to call `reset_lazy_props` afterwards to reflect
    the changes.
    """

    __slots__ = ("_target",)

    # lazy data calculated when first needed
    _target: typing.Optional[ProjectionTarget]

    def __init__(self, xyz: ArrayLike):
        xyz = array_chk(xyz, ((2, None), 3), chk_finite=True, dtype=float, copy=False)
        self.xyzd = np.empty((len(xyz), 4))
        self.xyz[...] = xyz

        self.dists[0] = 0
        segment_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        np.cumsum(segment_lengths, out=self.dists[1:])

        self.reset_lazy_props()

    @property
    def target(self) -> ProjectionTarget:
        if self._target is None:
            self._target = ProjectionTarget(self.xyz)
        return self._target

    @property
    def d_min(self) -> float:
        return float(self.xyzd[0, 3])

    @property
    def d_max(self) -> float:
        return float(self.xyzd[-1, 3])

    @property
    def length(self) -> float:
        return self.xyzd[-1, 3] - self.xyzd[0, 3]

    @classmethod
    def from_trusted_data(cls, xyzd: ArrayLike, copy: bool = True):
        """Create a Trajectory from existing coordinates and distances.

        No checks are performed.
        """
        trajectory = cls.__new__(cls)
        trajectory.xyzd = np.array(xyzd, dtype=float, copy=copy)
        trajectory.reset_lazy_props()
        return trajectory

    def reset_lazy_props(self):
        self._target = None

    def copy(self):
        return self.__class__.from_trusted_data(self.xyzd)

    def split(self, locations: Locations) -> list:
        """Split between locations."""
        if len(locations) < 2:
            raise ValueError("at least two locations are required")
        xyzds = substrings(self.xyzd, locations[:-1], locations[1:])
        return list(map(self.__class__.from_trusted_data, xyzds))

    def locate(self, dists: ArrayLike, extrapolate: bool = False) -> Locations:
        return locate(self.dists, np.asarray(dists, dtype=float), extrapolate)


class TrajectoryTrip(XYZDMixin):
    """A sequence of 3d points along a Trajectory with mandatory distance information.

    The distances are the travel distances along the trajectory in meters.
    All distances have to be finite, inside the distance range of the reference trajectory,
    and increasing in steps of at least `snapping_params.min_spacing`.

    `dists_trusted` is a Boolean array indicating whether the corresponding distance is accurate or
    merely a rough hint.

    `snapping_params` controls how snapping will be performed.
    The supplied distances have to pass the `self.dists_ok` test.

    Whenever you change `xyzd` or update the target of the trajectory, you have to call
    `reset_lazy_props` to reflect the changes.
    """

    __slots__ = (
        "trajectory",
        "dists_trusted",
        "snapping_params",
        "_line_fractions",
        "_projected_points",
    )

    trajectory: Trajectory
    dists_trusted: np.ndarray
    snapping_params: SnappingParams
    # lazy data calculated when first needed
    _line_fractions: typing.Optional[LineFractions]
    _projected_points: typing.Optional[ProjectedPoints]

    def __init__(
        self,
        trajectory: Trajectory,
        xyzd: ArrayLike,
        dists_trusted: ArrayLike,
        snapping_params: SnappingParams = DEFAULT_SNAPPING_PARAMS,
    ):
        self.trajectory = trajectory
        self.xyzd = array_chk(xyzd, (None, 4), chk_finite=True, dtype=float)
        self.dists_trusted = array_chk(dists_trusted, (len(self),), dtype=bool)
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
        self.reset_lazy_props()

    @classmethod
    def from_dubious_dists_and_times(
        cls,
        trajectory: Trajectory,
        xyzd: ArrayLike,
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
        trip = cls.__new__(cls)
        trip.trajectory = trajectory
        trip.xyzd = array_chk(xyzd, (None, 4), dtype=float)
        if not np.all(np.isfinite(trip.xyz)):
            raise ValueError("coords have to be finite")
        missing = np.isnan(trip.dists)
        available = np.logical_not(missing)
        trip.dists_trusted = available.copy()
        trip.snapping_params = snapping_params
        times = array_chk(times, (len(trip),), dtype=float, copy=False)
        # lazy props only depend on xyz and they are needed in _fix_distances,
        # so we reset them at this point
        trip.reset_lazy_props()

        trip._fix_distances(missing, available, times)

        # is our implementation working?
        assert np.all(np.isfinite(trip.dists))

        if snapping_params.short_trajectory_fallback and trip.trajectory_is_too_short:
            assert trip.dists_ok(trip.dists, min_spacing=0.0, atol=1e-3)
        else:
            assert trip.dists_ok(trip.dists)
        return trip

    @property
    def trajectory_is_too_short(self) -> bool:
        required_length = max(0, len(self) - 1) * self.snapping_params.min_spacing
        # use >= to trigger fallback also when min_spacing is 0 and length is 0
        return required_length >= self.trajectory.length

    @property
    def line_fractions(self) -> LineFractions:
        if self._line_fractions is None:
            self._line_fractions = self.trajectory.target.get_line_fractions(self.xyz)
        return self._line_fractions

    @property
    def projected_points(self) -> ProjectedPoints:
        """Trip points projected to the trajectory.

        This is only the eucledian projection to the entire trajectory,
        not necessarlily the snapping result.
        """
        if self._projected_points is None:
            self._projected_points = self.line_fractions.project()
        return self._projected_points

    def reset_lazy_props(self) -> None:
        self._line_fractions = None
        self._projected_points = None

    def snap_trip_points(
        self,
        convergence_accuracy: float = 1.0,
        n_iter_max: int = 1000,
    ) -> "SnappedTripPoints":
        """Snap trip points to the trajectory.

        Points with trusted distances end up exactly at those distances.
        Points with untrusted distances are placed acoording to untrusted distances
        and then iteratively optimized such that they get closer to
        the unsnapped point coords but stay on the trajectory and
        keep their order and minimum spacing intact.

        All distances are in meters.

        convergence_accuracy -- stop iterative solution when distances do not move more than this
        """

        snapped_points = ProjectedPoints.empty(len(self), 3)
        methods = np.empty_like(self.dists, dtype=object)
        distances = np.empty_like(self.dists)
        snapped_trip_points = SnappedTripPoints(
            self, snapped_points, methods, False, distances
        )

        trusted_indices = np.nonzero(self.dists_trusted)[0]
        reverse_order_allowed = (
            len(trusted_indices) == 0 and self.snapping_params.reverse_order_allowed
        )

        # trusted distances
        trusted_dists = self.dists[trusted_indices]
        trusted_locations = self.trajectory.locate(trusted_dists)
        snapped_points.locations[trusted_indices] = trusted_locations
        trusted_coords = interpolate(self.trajectory.xyz, trusted_locations)
        snapped_points.coords[trusted_indices] = trusted_coords
        snapped_points.cartesian_distances[trusted_indices] = np.linalg.norm(
            trusted_coords - self.xyz[trusted_indices], axis=1
        )
        methods[trusted_indices] = SnappingMethod.trusted
        distances[trusted_indices] = trusted_dists

        # every group of consecutive points with untrusted distances is an isolated problem
        untrusted_indices = np.nonzero(np.logical_not(self.dists_trusted))[0]
        for group_indices in iter_consecutive_groups(untrusted_indices):
            group_slice = slice(group_indices[0], group_indices[-1] + 1)
            self._snap_untrusted(
                snapped_trip_points,  # will be modified inplace
                group_slice=group_slice,
                reverse_order_allowed=reverse_order_allowed,
                convergence_accuracy=convergence_accuracy,
                n_iter_max=n_iter_max,
            )

        return snapped_trip_points

    def dists_ok(
        self,
        dists: ArrayLike,
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
            dists,
            d_min=d_min,
            d_max=d_max,
            min_spacing=min_spacing,
            atol=atol,
        )

    def _fix_distances(  # noqa: old code too complex (needs refactoring)
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
        shortest_dists = self.projected_points.cartesian_distances[available_indices]
        snapped_coords = interpolate(
            self.trajectory.xyz, self.trajectory.locate(self.dists[available_indices])
        )
        snapping_dists = np.linalg.norm(
            snapped_coords - self.xyz[available_indices], axis=1
        )
        keep_mask = self.snapping_params.keep_snapping_dists(
            snapping_dists, shortest_dists
        )
        trust_mask = self.snapping_params.trust_snapping_dists(
            snapping_dists, shortest_dists
        )
        # python loop for individual logging
        for i, keep, trust in zip(available_indices, keep_mask, trust_mask):
            if not keep:
                logger.debug("discarding distance %d: too far away from trip point", i)
                self.dists_trusted[i] = False
                missing[i] = True
                available[i] = False
                self.dists[i] = np.nan
            elif not trust:
                logger.debug("distrusting distance %d: too far away from trip point", i)
                self.dists_trusted[i] = False
        del available_indices  # invalid at this point, but not needed any more

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
                    logger.debug("fixing distance at %d to boundary", index)
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
            try:
                logger.debug("trying to estimate distances from times")
                y_prime = resample(
                    x=times[input_indices],
                    y=self.dists[input_indices],
                    x_prime=times[output_indices],
                    extrapolate=True,
                )
            except ExtrapolationError:
                logger.debug(
                    "estimating distances from times failed (not enough unique times)"
                )
                pass
            else:
                logger.debug(
                    "estimated %d/%d distances from times",
                    len(output_indices),
                    len(self),
                )
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
                logger.debug("fixing distance at %d to boundary", index)
                missing[index] = False
                available[index] = True
                self.dists[index] = value
                n_available += 1

        # shortcut if fixing the boundaries fixed everything
        if n_available == len(self):
            return

        assert n_available >= 2
        y_prime = resample(
            x=np.nonzero(available)[0].astype(float),
            y=self.dists[available],
            x_prime=np.nonzero(missing)[0].astype(float),
        )
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
            v_min = self.dists[i_left] + min_spacing if i_left >= 0 else d_min
            v_max = self.dists[i_right] - min_spacing if i_right < n_points else d_max
            # Due to tolerance in previous steps, it can happen in rare cases that
            # we run slightly out of space (still within tolerances). In that case, we just apply
            # equidistant placement.
            if v_max < v_min:
                if len(group_indices) > 1:
                    raise ValueError(
                        f"tolerance {atol} was too high for min_spacing {min_spacing}"
                    )
                v_mean = (v_max + v_min) / 2
                v_mean = min(max(v_mean, d_min), d_max)
                v_min = v_max = v_mean
            try:
                self.dists[group_indices] = fix_sequence_with_missing_values(
                    values=self.dists[group_indices],
                    v_min=v_min,
                    v_max=v_max,
                    d_min=min_spacing,
                    atol=atol,
                )
            except NoSolution:
                self.dists[group_indices] = np.linspace(
                    v_min, v_max, len(group_indices)
                )

    def _snap_untrusted(
        self,
        snapped_trip_points: "SnappedTripPoints",  # modified inplace
        group_slice: slice,
        reverse_order_allowed: bool,
        convergence_accuracy: float,
        n_iter_max: int,
    ) -> None:
        # calculate free space where we can snap to without getting too close
        # to points with trusted dists
        assert group_slice.start >= 0
        assert group_slice.stop > group_slice.start
        params = self.snapping_params
        left = group_slice.start - 1
        right = group_slice.stop
        if left < 0:
            d_min = self.trajectory.d_min
        else:
            d_min = self.dists[left] + params.min_spacing
        if right >= len(self):
            d_max = self.trajectory.d_max
        else:
            d_max = self.dists[right] - params.min_spacing
        n_points = group_slice.stop - group_slice.start

        # Due to tolerances in previous steps, we can run slightly out of space.
        # In that case, fall back to equidistant placement.
        if d_max < d_min:
            if n_points > 1:
                raise ValueError(
                    f"tolerance {params.atol_spacing} was too high "
                    f"for min_spacing {params.min_spacing}"
                )
            d_mean = (d_max + d_min) / 2
            d_mean = min(max(d_mean, self.trajectory.d_min), self.trajectory.d_max)
            d_min = d_max = d_mean
        available_length = d_max - d_min
        required_length = (n_points - 1) * params.min_spacing
        if required_length > available_length:
            logger.debug("fallback for slightly not enough space")
            method = SnappingMethod.fallback
            if reverse_order_allowed:
                # this is such a rare case that we don't make the effort
                # to check the reverse solution
                logger.warning("ignoring possible reverse solution in fallback")
            distances = np.linspace(d_min, d_max, n_points)
            locations = self.trajectory.locate(distances)
            coords = interpolate(self.trajectory.xyz, locations)
            snapped_points = ProjectedPoints(
                coords,
                locations,
                np.linalg.norm(coords - self.xyz[group_slice], axis=1),
            )
        else:
            # first try just projecting; if this works, we're done
            snapped_points = self.line_fractions[group_slice].project_between_distances(
                d_min, d_max, self.trajectory.dists
            )
            distances = interpolate(self.trajectory.dists, snapped_points.locations)

            d_ok = partial(self.dists_ok, d_min=d_min, d_max=d_max)
            if d_ok(distances):
                logger.debug("projecting in forward direction succesful")
                method = SnappingMethod.projected
            elif reverse_order_allowed and d_ok(distances[::-1]):
                logger.debug("projecting in reverse direction succesful")
                method = SnappingMethod.projected
                snapped_trip_points.reverse_order = True
            else:
                logger.debug("projected solution not admissible")
                method = SnappingMethod.iterative
                snapped_points, distances = self._snap_untrusted_iteratively(
                    group_slice=group_slice,
                    d_min=d_min,
                    d_max=d_max,
                    reverse=False,
                    convergence_accuracy=convergence_accuracy,
                    n_iter_max=n_iter_max,
                )
                logger.debug("forward distances: %s", distances)
                if reverse_order_allowed:
                    # pick forward/backward solution with better sum of squared snapping distances
                    residuum = (snapped_points.cartesian_distances**2).sum()
                    (
                        alt_snapped_points,
                        alt_distances,
                    ) = self._snap_untrusted_iteratively(
                        group_slice=group_slice,
                        d_min=d_min,
                        d_max=d_max,
                        reverse=True,
                        convergence_accuracy=convergence_accuracy,
                        n_iter_max=n_iter_max,
                    )
                    logger.debug("backward distances: %s", alt_distances)
                    alt_residuum = (alt_snapped_points.cartesian_distances**2).sum()
                    logger.debug("residuum: %s", residuum)
                    logger.debug("backward residuum: %s", alt_residuum)
                    if alt_residuum < residuum:
                        snapped_trip_points.reverse_order = True
                        snapped_points = alt_snapped_points
                        distances = alt_distances

        snapped_trip_points.distances[group_slice] = distances
        snapped_trip_points.snapped_points[group_slice] = snapped_points
        snapped_trip_points.methods[group_slice] = method

    def _snap_untrusted_iteratively(
        self,
        group_slice: slice,
        d_min: float,
        d_max: float,
        reverse: bool,
        convergence_accuracy: float,
        n_iter_max: int,
    ) -> typing.Tuple[ProjectedPoints, np.ndarray]:
        """Snap points with untrusted distances iteratively.

        Start with an admissible initial guess for the snapped points.
        Then calculate a region around each snapped point where the point
        is allowed to be such that all points can be placed anywhere inside
        their region without coming too close to any of the other regions or the
        boundaries. Place each snapped point closest to its source point inside its region
        (or in other words: project the source point to the region).
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

        If `reverse` is set to `True`, the order of the points is reversed internally
        and the initial distances are reflected at `(d_min + d_max) / 2` and reversed.

        The order of the output is always consistent with the order of `group_slice` (not reversed).
        """
        assert group_slice.start >= 0
        assert group_slice.stop > group_slice.start
        half_min_spacing = 0.5 * self.snapping_params.min_spacing
        n_points = group_slice.stop - group_slice.start
        snapped_points = ProjectedPoints.empty(n_points, 3)
        if reverse:
            group_slice = slice(
                group_slice.stop - 1,
                group_slice.start - 1 if group_slice.start > 0 else None,
                -1,
            )
            distances = self.dists[group_slice]
            distances = (d_min + d_max) - distances
            # a view of snapped_points consistent with increasing distance order
            sp_view = snapped_points[::-1]
        else:
            distances = self.dists[group_slice].copy()
            sp_view = snapped_points

        if not self.dists_ok(distances, d_min=d_min, d_max=d_max):
            raise ValueError("bad initial untrusted point distances")

        # centers of forbidden regions for the next round
        separators = np.empty(n_points + 1)
        separators[0] = d_min - half_min_spacing
        separators[1:-1] = 0.5 * (distances[:-1] + distances[1:])
        separators[-1] = d_max + half_min_spacing

        # regions for the next round
        regions = np.empty((n_points, 2))

        # regions where we last projected to
        projected_regions = np.full_like(regions, np.nan)

        n_iter = 0
        # maximum distances change compared to last iteration
        delta = float("inf")

        # use slices, not index lists to get views instead of copies
        # (we write to region_points later and want this to be reflected in snapped_points)
        region_line_fractions = [
            self.line_fractions[i : i + 1] for i in range(len(self))[group_slice]
        ]
        rprange = range(n_points - 1, -1, -1) if reverse else range(n_points)
        region_points = [snapped_points[i : i + 1] for i in rprange]

        global_mins = self.projected_points.cartesian_distances[group_slice]
        optimal = np.zeros(n_points, dtype=bool)

        # close to convergence, we move half the last delta in the next step, so if we would
        # run forever, we would still move (delta * sum (1/2 ** n), n=1 to infinity) = delta,
        # thus in all following steps combined, we move at most delta
        # so `convergence_accuracy / 1.1` should be fine to detect convergence with desired accuracy
        while delta > convergence_accuracy / 1.1:
            n_iter += 1
            if n_iter > n_iter_max:
                raise SnappingError("not converged")

            regions[:, 0] = separators[:-1] + half_min_spacing
            regions[:, 1] = separators[1:] - half_min_spacing
            reproject = np.nonzero(
                np.logical_not(optimal) & np.any(regions != projected_regions, axis=1)
            )[0]
            if not len(reproject):
                break

            # TODO (nice to have / performance): we already projected globally
            # so we could check for each point if the global distance is inside the region,
            # then we could take the global solution for this point instead of projecting again
            for i in reproject:
                logger.debug(
                    "iter %d: reprojecting %d to [%.3f km, %.3f km]",
                    n_iter,
                    i,
                    *(regions[i] * 1e-3),
                )
                # we write to the proper place in snapped_points via region_points[i]
                region_line_fractions[i].project_between_distances(
                    *regions[i], self.trajectory.dists, out=region_points[i]
                )

            reprojected_distances = interpolate(
                self.trajectory.dists, sp_view.locations[reproject]
            )
            delta = np.abs(reprojected_distances - distances[reproject]).max()
            distances[reproject] = reprojected_distances

            # We have a greedy algorithm, so once we reach the global optimum for a point,
            # we will never give it up again. So we can quickly move the separators
            # to speed up convergence of the other points.
            optimal[reproject] = (
                np.abs(sp_view.cartesian_distances[reproject] - global_mins[reproject])
                <= convergence_accuracy
            )

            optimal_indices = np.nonzero(optimal)[0]
            if len(optimal_indices) == n_points:
                break

            # for simplicity we first calculate default separators, then overwrite
            # to avoid fiddling with more masks or indices
            separators[1:-1] = 0.5 * (distances[:-1] + distances[1:])
            optimal_distances = distances[optimal_indices]
            separators[optimal_indices] = optimal_distances - half_min_spacing
            separators[optimal_indices + 1] = optimal_distances + half_min_spacing

        logger.debug("converged in %d iterations", n_iter)

        assert self.dists_ok(
            distances, d_min=d_min, d_max=d_max
        ), "converged dists not admissible"

        if reverse:
            distances = distances[::-1]

        return snapped_points, distances


class SnappedTripPoints:
    __slots__ = ("trip", "snapped_points", "distances", "methods", "reverse_order")

    trip: TrajectoryTrip
    snapped_points: ProjectedPoints
    distances: np.ndarray
    methods: np.ndarray
    reverse_order: bool

    def __init__(
        self,
        trip: TrajectoryTrip,
        snapped_points: ProjectedPoints,
        methods: ArrayLike,
        reverse_order: bool,
        distances: typing.Optional[np.ndarray] = None,
    ):
        if len(snapped_points) != len(trip):
            raise ValueError("wrong number of snapped points for trip")
        self.trip = trip
        self.snapped_points = snapped_points
        self.methods = array_chk(methods, (len(trip),), dtype=object, copy=False)
        self.reverse_order = reverse_order
        if distances is None:
            self.distances = interpolate(
                trip.trajectory.dists, snapped_points.locations
            )
        else:
            self.distances = distances

    @property
    def snapping_distances(self) -> np.ndarray:
        """The distances between trip points and snapped trip points in meters."""
        return self.snapped_points.cartesian_distances

    @property
    def shortest_distances(self) -> np.ndarray:
        """The shortest distances between trip points and the trajectory in meters."""
        return self.trip.projected_points.cartesian_distances

    def snapping_distances_valid(
        self, rtol: typing.Optional[float] = None, atol: typing.Optional[float] = None
    ) -> np.ndarray:
        if rtol is None:
            rtol = self.trip.snapping_params.rtol_keep
        if atol is None:
            atol = self.trip.snapping_params.atol_keep
        return self.trip.snapping_params.snapping_dists_admissible(
            self.snapping_distances, self.shortest_distances, rtol=rtol, atol=atol
        )

    def raise_invalid(
        self,
        rtol: typing.Optional[float] = None,
        atol: typing.Optional[float] = None,
        debug_geojson_path: typing.Optional[str] = None,
    ):
        """Check if all snapping distances are valid and raise SnappingError if not valid.

        Default tolerances are *tol_keep from `self.trip.snapping_params`.

        Includes debug GeoJSON in the error message.
        Optionally dumps debug GeoJSON to a file.
        """
        valid = self.snapping_distances_valid(rtol=rtol, atol=atol)
        if not np.all(valid):
            debug = json.dumps(self.to_geojson(valid), indent=2)
            if debug_geojson_path:
                logger.info("writing debug GeoJSON to %r", debug_geojson_path)
                with open(debug_geojson_path, "w") as handle:
                    handle.write(debug)
            raise SnappingError(
                f"snapped point too far away from original point:\n{debug}\n"
            )

    def get_inter_point_trajectories(self) -> typing.List[Trajectory]:
        """Split the trajectory at the snapped points.

        Return sub-trajectories between consecutive pairs of trip points
        in the same order as in the trip.

        Attention!
        Sub-trajectories are always oriented in the same direction
        as the original trajectory. So if `self.reverse_order` is `True`,
        all sub-trajectories will run from destination point to source point.
        If you only need the coordinates (and not the distances) of the trajectories but oriented
        in travel direction, use `get_inter_point_ls_coords_in_travel_direction` instead.
        """
        trajectory = self.trip.trajectory
        if self.reverse_order:
            return trajectory.split(self.snapped_points.locations[::-1])[::-1]
        else:
            return trajectory.split(self.snapped_points.locations)

    def get_inter_point_ls_coords_in_travel_direction(
        self,
        crs: pyproj.CRS = EPSG4978,
        strict_axis_order: bool = False,
        without_z: bool = False,
    ) -> typing.List[np.ndarray]:
        """Split the trajectory at the snapped points.

        Return linestring coords between consecutive pairs of trip points
        in the same order as in the trip.
        All linestrings are oriented in travel direction of the trip.

        `crs`, `strict_axis_order` and `without_z` control the optional transformation
        applied to the coords.
        """
        trajectories = self.get_inter_point_trajectories()

        if crs is EPSG4978 or crs == EPSG4978:
            if without_z:

                def trafo(coords):
                    return coords[:, :2].copy()

            else:

                def trafo(coords):
                    return coords.copy()

        else:
            trafo = partial(
                transform_coords,
                trafo=get_trafo(EPSG4978, crs, strict_axis_order),
                skip_z_output=without_z,
            )

        if self.reverse_order:
            return [trafo(t.xyz[::-1]) for t in trajectories]
        else:
            return [trafo(t.xyz) for t in trajectories]

    def get_inter_point_ls_lon_lat_in_travel_direction(self):
        """Convenience method to get WGS84 longitude,latitude linestring coords after splitting."""
        return self.get_inter_point_ls_coords_in_travel_direction(
            crs=EPSG4326, strict_axis_order=False, without_z=True
        )

    def to_geojson(
        self,
        d_valids: typing.Optional[np.ndarray] = None,
        initial_locations: typing.Optional[Locations] = None,
    ) -> typing.Dict[str, typing.Any]:
        if d_valids is None:
            d_valids = self.trip.snapping_params.keep_snapping_dists(
                self.snapping_distances, self.shortest_distances
            )
        trajectories = self.get_inter_point_trajectories()
        trafo = partial(
            transform_coords,
            trafo=get_trafo(EPSG4978, EPSG4326, strict_axis_order=False),
        )

        # whole trajectory
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": trafo(self.trip.trajectory.xyz).tolist(),
                },
                "properties": {
                    "label": "full trajectory",
                    "from_index": 0,
                    "to_index": len(self.trip) - 1,
                    "length": float(self.trip.trajectory.length),
                },
            }
        ]

        # splitted trajectory
        features.extend(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": trafo(t.xyz).tolist(),
                },
                "properties": {
                    "label": "partial trajectory",
                    "from_index": i,
                    "to_index": i + 1,
                    "length": float(t.length),
                },
            }
            for i, t in enumerate(trajectories)
        )

        # trip points
        trip_point_coords = trafo(self.trip.xyz)
        features.extend(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coords.tolist(),
                },
                "properties": {
                    "label": "trip point",
                    "index": i,
                    "dist_to_trip_point": 0.0,
                    "dist_along_traj": dist,
                    "method": method.value,
                    "valid": bool(valid),
                },
            }
            for i, (coords, dist, valid, method) in enumerate(
                zip(
                    trip_point_coords,
                    self.trip.dists,
                    d_valids,
                    self.methods,
                )
            )
        )

        features.extend(
            self._projected_points_to_features(
                self.trip.projected_points,
                "projected point",
                trip_point_coords,
                d_valids,
                interpolate(
                    self.trip.trajectory.dists, self.trip.projected_points.locations
                ),
                trafo,
            )
        )
        features.extend(
            self._projected_points_to_features(
                self.snapped_points,
                "snapped point",
                trip_point_coords,
                d_valids,
                self.distances,
                trafo,
            )
        )

        if initial_locations is not None:
            initial_coords = interpolate(self.trip.trajectory.xyz, initial_locations)
            initial_points = ProjectedPoints(
                initial_coords,
                initial_locations,
                np.linalg.norm(self.trip.xyz - initial_coords, axis=1),
            )
            features.extend(
                self._projected_points_to_features(
                    initial_points,
                    "initial point",
                    trip_point_coords,
                    d_valids,
                    interpolate(self.trip.trajectory.dists, initial_locations),
                    trafo,
                )
            )

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {"trajectory_reversed": bool(self.reverse_order)},
        }

    def _projected_points_to_features(
        self,
        points: ProjectedPoints,
        label: str,
        transformed_trip_point_coords: np.ndarray,
        d_valids: np.ndarray,
        dists_along_traj: np.ndarray,
        trafo,
    ) -> typing.Generator[dict, None, None]:
        tcoords = trafo(points.coords)

        # point features
        yield from (
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coords.tolist(),
                },
                "properties": {
                    "label": label,
                    "index": i,
                    "dist_to_trip_point": float(dist_to_tp),
                    "dist_along_traj": float(dist_along_traj),
                    "method": method.value,
                    "valid": bool(d_valid),
                },
            }
            for i, (coords, dist_to_tp, dist_along_traj, method, d_valid,) in enumerate(
                zip(
                    tcoords,
                    points.cartesian_distances,
                    dists_along_traj,
                    self.methods,
                    d_valids,
                )
            )
        )
        # linestring features from trip points to points (for drawing arrows)
        yield from (
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        from_coords.tolist(),
                        to_coords.tolist(),
                    ],
                },
                "properties": {
                    "label": f"{label} arrow",
                    "index": i,
                    "length": float(length),
                    "method": method.value,
                    "valid": bool(l_valid),
                },
            }
            for i, (from_coords, to_coords, length, method, l_valid) in enumerate(
                zip(
                    transformed_trip_point_coords,
                    tcoords,
                    points.cartesian_distances,
                    self.methods,
                    d_valids,
                )
            )
        )
