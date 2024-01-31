import typing
from functools import partial
import logging

import pyproj
import numpy as np
from numpy.typing import ArrayLike

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

    If an external location is trusted, snapping will occur exactly to this location. In
    between trusted locations, points will be snapped minimizing the sum of square
    distances with the given boundary conditions (individual maximum snapping radii for
    the points, minimum spacing along the trajectory).

    All distances are in meters.

    `min_spacing`:
    The minimum travel distance along the trajectory between consecutive snapped points.
    If external locations violate the minimum spacing (including the condition to
    provide enough space for points without an external location), all external
    locations will be distrusted.

    `rtol_trusted`, `atol_trusted`:
    Let `d_min` be the minimum distance of the point to the trajectory and `d` the
    distance of the point to the external location on the trajectory. If `d <=
    rtol_trusted * d_min + atol_trusted`, the external location is trusted (if not
    untrusted by other conditions).

    `rtol_snap`, `atol_snap`:
    Let `d_min` be as explained above. Then snapping between trusted locations is
    attempted to all parts of the trajectory that are within a radius of `rtol_snap *
    d_min + atol_snap`.

    `reverse_order_allowed`:
    Whether to also try reversing the trajectory. Only applies if no external locations
    are trusted.
    """

    min_spacing: float = 25.0
    rtol_trusted: float = 1.5
    atol_trusted: float = 10.0
    rtol_snap: float = 2.5
    atol_snap: float = 300.0
    reverse_order_allowed: bool = True

    def spacing_ok(
        self,
        values: ArrayLike,
        d_min: float,
        d_max: float,
    ) -> bool:
        """Check if all non-NaN values are in [d_min, d_max] with the correct spacing.

        Spacing for NaN values is also considered. For example: If there are three NaN
        values between two non-NaN values, the spacing between the non-NaN values has to
        be at least `4 * self.min_spacing`.
        """
        if d_max < d_min:
            raise ValueError(
                f"d_max={d_max} has to be greater than or equal to d_min={d_min}"
            )

        values_arr = np.asarray(values, dtype=float)

        # padding to treat NaNs at the boundary, d_min, d_max and self.min_spacing in an
        # elegant way
        padded_values = np.empty(len(values_arr) + 2)
        padded_values[0] = d_min - self.min_spacing
        padded_values[1:-1] = values_arr
        padded_values[-1] = d_max + self.min_spacing
        del values_arr  # prevent accidental usage

        non_nan_indices = np.nonzero(np.logical_not(np.isnan(padded_values)))[0]
        min_spacing_arr = np.diff(non_nan_indices).astype(float)
        min_spacing_arr *= self.min_spacing

        # bool to ensure builtin bool (and not numpy bool)
        return bool(np.all(np.diff(padded_values[non_nan_indices]) >= min_spacing_arr))

    def snapping_dists_trusted(
        self,
        snapping_dists: ArrayLike,
        shortest_dists: ArrayLike,
    ) -> np.ndarray:
        """Get a boolean mask indicating which snapping distances can be trusted."""
        snapping_dists = np.asarray(snapping_dists, dtype=float)
        shortest_dists = np.asarray(shortest_dists, dtype=float)
        return snapping_dists <= self.rtol_trusted * shortest_dists + self.atol_trusted


DEFAULT_SNAPPING_PARAMS = SnappingParams()


class DubiousTrajectory(XYZDMixin):
    """A linestring with dubious travel distances for its vertices.

    The distances are to be interpreted as travel distances along the linestring in an
    arbitrary unit with arbitrary offset.

    Distances are allowed to contain NaNs and they are allowed to decrease (thus
    "dubious").

    The interpretation of the coordinates is given by `crs` (default: epsg:4326). By
    default, traditional GIS order is used for the axis order, that is longitude,
    latitude for geographic CRS and easting, northing for most projected CRS. If you
    want strict axis order as defined by the CRS, set `strict_axis_order` to True.
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
    ) -> None:
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


class DubiousTrajectoryTrip(XYZDMixin):
    """A sequence of points along a `DubiousTrajectory` with dubious distances.

    The interpretation of the coordinates is given by `trajectory.crs`. Points do not
    have to lie on the trajectory.

    The distances are to be interpreted as in `trajectory`. Distances are allowed to
    contain NaNs. They also may fall outside the range of the distances of the
    trajectory and they are allowed to decrease.

    The order of the points has to match the canonical travel direction of the
    trajectory. Exception: If all distances are missing, the points are allowed to be in
    reverse order.
    """

    __slots__ = ("trajectory",)

    trajectory: DubiousTrajectory

    def __init__(self, trajectory: DubiousTrajectory, xyzd: ArrayLike) -> None:
        self.trajectory = trajectory
        self.xyzd = array_chk(xyzd, (None, 4), dtype=float)
        if not np.all(np.isfinite(self.xyz)):
            raise ValueError("coords have to be finite")

    def to_trajectory_trip(self) -> "TrajectoryTrip":
        """Convert to a meter based `TrajectoryTrip`.

        Conversion of trip distances is only possible if there are at least two finite
        distances and all finite distances are monotonically non-decreasing. If this is
        not the case, the metric trip distances will all be set to NaN.
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

        return TrajectoryTrip(trajectory=traj, xyzd=trip_xyzd)


_TVTrajectory = typing.TypeVar("_TVTrajectory", bound="Trajectory")


class Trajectory(XYZDMixin):
    """A 3d linestring in epsg:4978 with metric travel distances.

    Distances are the cumulative cartesian distances along the trajectory in meters. As
    long as segments are not too long (say a few 100 km per segment), the distances are
    a very good approximation to WGS84 geodesic distances (including altitude changes).
    Distances do not have to start at 0. An arbitrary offset is allowed. This is useful
    for relating parts of a trajectory to the original trajectory.

    Whenever you make changes to `xyzd`, you have to call `reset_lazy_props` afterwards
    to reflect the changes.
    """

    __slots__ = ("_target",)

    # lazy data calculated when first needed
    _target: typing.Optional[ProjectionTarget]

    def __init__(self, xyz: ArrayLike) -> None:
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
    def from_trusted_data(
        cls: type[_TVTrajectory], xyzd: ArrayLike, copy: bool = True
    ) -> _TVTrajectory:
        """Create a Trajectory from existing coordinates and distances.

        No checks are performed.
        """
        trajectory = cls.__new__(cls)
        trajectory.xyzd = np.array(xyzd, dtype=float, copy=copy)
        trajectory.reset_lazy_props()
        return trajectory

    def reset_lazy_props(self) -> None:
        self._target = None

    def copy(self: _TVTrajectory) -> _TVTrajectory:
        return self.__class__.from_trusted_data(self.xyzd)

    def split(self: _TVTrajectory, locations: Locations) -> list[_TVTrajectory]:
        """Split between locations."""
        if len(locations) < 2:
            raise ValueError("at least two locations are required")
        xyzds = substrings(self.xyzd, locations[:-1], locations[1:])
        return list(map(self.__class__.from_trusted_data, xyzds))

    def locate(self, dists: ArrayLike, extrapolate: bool = False) -> Locations:
        return locate(self.dists, np.asarray(dists, dtype=float), extrapolate)


class TrajectoryTrip(XYZDMixin):
    """A sequence of 3d points along a `Trajectory` with optional distance information.

    The distances are the travel distances along the trajectory in meters. Distances are
    allowed to contain NaNs. They also may fall outside the range of the distances of
    the trajectory and they are allowed to decrease.

    Whenever you change `xyzd` or update the target of the trajectory, you have to call
    `reset_lazy_props` to reflect the changes.
    """

    __slots__ = (
        "trajectory",
        "_line_fractions",
        "_projected_points",
    )

    trajectory: Trajectory

    # lazy data calculated when first needed
    _line_fractions: typing.Optional[LineFractions]
    _projected_points: typing.Optional[ProjectedPoints]

    def __init__(self, trajectory: Trajectory, xyzd: ArrayLike) -> None:
        self.trajectory = trajectory
        self.xyzd = array_chk(xyzd, (None, 4), dtype=float)
        if not np.all(np.isfinite(self.xyz)):
            raise ValueError("coords have to be finite")
        self.reset_lazy_props()

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

    def snap_trip_points(self, params: SnappingParams) -> "SnappedTripPoints":
        """Snap trip points to the trajectory.

        See `SnappingParams` for an explanation how the snapping is done and how it can
        be controlled.
        """
        snapped_trip_points, trusted_mask = self._snap_trusted(params)
        reverse_order_allowed = bool(
            params.reverse_order_allowed and np.all(np.isnan(self.dists))
        )

        # every group of consecutive points with untrusted distances is an isolated
        # problem
        untrusted_indices = np.nonzero(np.logical_not(trusted_mask))[0]
        for group_indices in iter_consecutive_groups(untrusted_indices):
            group_slice = slice(group_indices[0], group_indices[-1] + 1)
            self._snap_untrusted(
                snapped_trip_points,  # will be modified inplace
                group_slice=group_slice,
                reverse_order_allowed=reverse_order_allowed,
                params=params,
            )

        return snapped_trip_points

    def _snap_trusted(
        self, params: SnappingParams
    ) -> tuple["SnappedTripPoints", np.ndarray]:
        snapped_points = ProjectedPoints.empty(len(self), 3)
        methods = np.empty_like(self.dists, dtype=object)
        distances = np.empty_like(self.dists)
        snapped_trip_points = SnappedTripPoints(
            self, snapped_points, methods, False, distances
        )

        traj_spacing_ok = partial(
            params.spacing_ok, d_min=self.trajectory.d_min, d_max=self.trajectory.d_max
        )
        if traj_spacing_ok(self.dists):
            trusted_mask = np.logical_not(np.isnan(self.dists))
            n_non_nan = trusted_mask.sum()
            logger.debug(
                "spacing is OK for %d distances (%d points)", n_non_nan, len(self)
            )
            dists = self.dists[trusted_mask]
            locations = self.trajectory.locate(dists)
            coords = interpolate(self.trajectory.xyz, locations)
            snapping_dists = np.linalg.norm(coords - self.xyz[trusted_mask], axis=1)
            trusted_sub_mask = params.snapping_dists_trusted(
                snapping_dists,
                self.projected_points.cartesian_distances[trusted_mask],
            )
            logger.debug(
                "snapping distance is OK for %d / %d points with a distance",
                trusted_sub_mask.sum(),
                n_non_nan,
            )
            trusted_mask[trusted_mask] = trusted_sub_mask
            snapped_points.coords[trusted_mask] = coords[trusted_sub_mask]
            snapped_points.cartesian_distances[trusted_mask] = snapping_dists[
                trusted_sub_mask
            ]
            snapped_points.locations[trusted_mask] = locations[trusted_sub_mask]
            methods[trusted_mask] = SnappingMethod.trusted
            distances[trusted_mask] = dists[trusted_sub_mask]
            # if the order was ok it is still ok when distrusting more points, so we are
            # done here
        else:
            logger.debug("spacing of distances is not OK, untrusting all distances")
            trusted_mask = np.zeros_like(self.dists, dtype=bool)
            # if the order was not ok, it can still be bad after untrusting all points
            # since the total length can be too short for the points
            fake_dists = np.full_like(self.dists, np.NaN)
            if not traj_spacing_ok(fake_dists):
                raise SnappingError(
                    "trajectory of length %g is too short for %d points with "
                    "min_spacing of %g",
                    self.trajectory.length,
                    len(self),
                    params.min_spacing,
                )

        return snapped_trip_points, trusted_mask

    def _snap_untrusted(
        self,
        snapped_trip_points: "SnappedTripPoints",  # modified inplace
        group_slice: slice,
        reverse_order_allowed: bool,
        params: SnappingParams,
    ) -> None:
        # calculate free space where we can snap to without getting too close
        # to points with trusted dists
        assert group_slice.start >= 0
        assert group_slice.stop > group_slice.start
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
        assert not np.any(np.isnan([d_min, d_max]))

        raise NotImplementedError("the new snapping method is not implemented yet")

        # setting this flag is well defined since when it is true, this implies that the
        # group spans the whole trajectory and then this method is called only once
        # snapped_trip_points.reverse_order = reverse_order

        # snapped_trip_points.distances[group_slice] = distances
        # snapped_trip_points.snapped_points[group_slice] = snapped_points
        snapped_trip_points.methods[group_slice] = SnappingMethod.projected


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

    def get_inter_point_trajectories(self) -> list[Trajectory]:
        """Split the trajectory at the snapped points.

        Return sub-trajectories between consecutive pairs of trip points in the same
        order as in the trip.

        Attention!
        Sub-trajectories are always oriented in the same direction as the original
        trajectory. So if `self.reverse_order` is `True`, all sub-trajectories will run
        from destination point to source point. If you only need the coordinates (and
        not the distances) of the trajectories but oriented in travel direction, use
        `get_inter_point_ls_coords_in_travel_direction` instead.
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
    ) -> list[np.ndarray]:
        """Split the trajectory at the snapped points.

        Return linestring coords between consecutive pairs of trip points in the same
        order as in the trip. All linestrings are oriented in travel direction of the
        trip.

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
