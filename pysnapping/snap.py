import typing
from functools import partial
from bisect import bisect_right
from operator import itemgetter
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
    get_trafo,
    transform_coords,
    array_chk,
    simplify_2d_keep_rest,
    cumulative_min_and_argmin,
)
from . import (
    SnappingError,
    BadShortestDistances,
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
    snapping distances with the given boundary conditions (individual maximum snapping
    radii for the points, minimum spacing along the trajectory).

    All distances are in meters.

    `min_spacing`:
    The minimum travel distance along the trajectory between consecutive snapped points.
    If external locations violate the minimum spacing (including the condition to
    provide enough space for points without an external location), all external
    locations will be distrusted.

    `rtol_trusted`, `atol_trusted`:
    Let `d_min` be the shortest distance of the point to the trajectory and `d` the
    distance of the point to the external location on the trajectory. If `d <=
    rtol_trusted * d_min + atol_trusted`, the external location is trusted (if not
    untrusted by other conditions).

    `rtol_snap`, `atol_snap`:
    Let `d_min` be as explained above. Then snapping between trusted locations is
    attempted to all parts of the trajectory that are within a radius of `rtol_snap *
    d_min + atol_snap`.

    `max_shortest_distance`
    The maximum shortest distance between any trip point and the trajectory. If this is
    exceeded, snapping is not attempted at all.

    `sampling_step`:
    The sampling distance step along the trajectory for candidate snapping targets.

    `reverse_order_allowed`:
    Whether to also try reversing the trajectory. Only applies if no external locations
    are given. The solution with minimum sum of square snapping distances will be
    applied.

    `short_trajectory_fallback`:
    If enabled and the trajectory is too short to snap all points with `min_spacing` and
    a numerical tolerance considering `sampling_step`, points will be placed
    equidistantly along the whole trajectory.
    """

    min_spacing: float = 25.0
    rtol_trusted: float = 1.5
    atol_trusted: float = 10.0
    rtol_snap: float = 2.5
    atol_snap: float = 300.0
    max_shortest_distance: float = 500.0
    sampling_step: float = 5.0
    reverse_order_allowed: bool = True
    short_trajectory_fallback: bool = True

    def spacing_ok(
        self,
        values: ArrayLike,
        d_min: float,
        d_max: float,
        consider_sampling_accuracy: bool = False,
    ) -> bool:
        """Check if all non-NaN values are in [d_min, d_max] with the correct spacing.

        Spacing for NaN values is also considered. For example: If there are three NaN
        values between two non-NaN values, the spacing between the non-NaN values has to
        be at least `4 * self.min_spacing`.

        If `consider_sampling_accuracy` is set to `True`, an extra spacing of `2.01 *
        sampling_step` is considered. This will ensure that the numerical solution can
        be found if the mathematical solution exists.
        """
        if d_max < d_min:
            raise ValueError(
                f"d_max={d_max} has to be greater than or equal to d_min={d_min}"
            )

        values_arr = np.asarray(values, dtype=float)

        min_spacing = self.min_spacing
        if consider_sampling_accuracy:
            min_spacing += 2.01 * self.sampling_step

        # padding to treat NaNs at the boundary, d_min, d_max and min_spacing in an
        # elegant way
        padded_values = np.empty(len(values_arr) + 2)
        padded_values[0] = d_min - min_spacing
        padded_values[1:-1] = values_arr
        padded_values[-1] = d_max + min_spacing
        del values_arr  # prevent accidental usage

        non_nan_indices = np.nonzero(np.logical_not(np.isnan(padded_values)))[0]
        min_spacing_arr = np.diff(non_nan_indices).astype(float)
        min_spacing_arr *= min_spacing

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

    def get_max_snapping_dists(self, shortest_dists: ArrayLike) -> np.ndarray:
        shortest_dists = np.asarray(shortest_dists, dtype=float)
        return self.rtol_snap * shortest_dists + self.atol_snap


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
            logger.debug("discarding trip distances (could not resample)")
            trip_dists.fill(np.nan)
        else:
            logger.debug("resampled trip distances")

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

    def snap_trip_points(
        self, params: SnappingParams = DEFAULT_SNAPPING_PARAMS
    ) -> "SnappedTripPoints":
        """Snap trip points to the trajectory.

        See `SnappingParams` for an explanation how the snapping is done and how it can
        be controlled.
        """
        bad_indices = np.nonzero(
            self.projected_points.cartesian_distances > params.max_shortest_distance
        )[0]
        if bad_indices.size:
            raise BadShortestDistances(
                bad_indices,
                self.projected_points.cartesian_distances,
                params.max_shortest_distance,
            )
        fallback = False
        reverse_order_allowed = params.reverse_order_allowed and bool(
            np.all(np.isnan(self.dists))
        )
        logger.debug("reverse order allowed: %s", reverse_order_allowed)
        try:
            trusted_ppoints_list = self.snap_trusted(params)
        except SnappingError:
            if params.short_trajectory_fallback:
                methods = np.full(len(self), SnappingMethod.fallback, dtype=object)
                ppoints_list = self.snap_fallback(params, reverse_order_allowed)
                params = params._replace(min_spacing=0)
                fallback = True
            else:
                raise
        else:
            methods = np.array(
                [
                    SnappingMethod.trusted
                    if ppoints is not None
                    else SnappingMethod.routed
                    for ppoints in trusted_ppoints_list
                ],
                dtype=object,
            )
            ppoints_list = self.set_untrusted_candidates_inplace(
                params, trusted_ppoints_list
            )

        # list of (ppoints, cost, reverse_order) tuples for forward and backward
        # solution
        results: list[tuple[ProjectedPoints, float, bool]] = []
        forward_error: typing.Optional[SnappingError] = None
        try:
            forward_result = self.route_candidates(params, ppoints_list) + (False,)
        except SnappingError as error:
            forward_error = error
            if not reverse_order_allowed:
                assert not fallback, "fallback solution should always be found"
                raise
        else:
            results.append(forward_result)

        if reverse_order_allowed:
            try:
                reversed_ppoints, reversed_cost = self.route_candidates(
                    params, ppoints_list[::-1]
                )
            except SnappingError as backward_error:
                if not results:
                    assert not fallback, "fallback solution should always be found"
                    raise backward_error from forward_error
            else:
                results.append((reversed_ppoints[::-1], reversed_cost, True))

        ppoints, cost, reverse_order = min(results, key=itemgetter(1))

        rms_residuum = (cost / len(self)) ** 0.5

        logger.debug(
            "Found solution with RMS residuum of %g meters. Reversed: %s. Fallback: %s",
            rms_residuum,
            reverse_order,
            fallback,
        )

        return SnappedTripPoints(
            trip=self,
            snapped_points=ppoints,
            methods=methods,
            reverse_order=reverse_order,
        )

    def snap_trusted(
        self, params: SnappingParams
    ) -> list[typing.Optional[ProjectedPoints]]:
        """Determine which points are trusted and snap them to the trajectory.

        See `SnappingParams` for an explanation of trusted/untrusted. The list contains
        `None` for each untrusted point and a `ProjectedPoints` with exactly one point
        for each trusted point.
        """
        traj_spacing_ok = partial(
            params.spacing_ok,
            d_min=self.trajectory.d_min,
            d_max=self.trajectory.d_max,
            consider_sampling_accuracy=True,
        )
        ppoints_list: list[typing.Optional[ProjectedPoints]] = [None] * len(self)
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
                "snapping distance is trusted for %d / %d points with a distance",
                trusted_sub_mask.sum(),
                n_non_nan,
            )
            trusted_mask[trusted_mask] = trusted_sub_mask
            # If the order was ok in the first place, it is still ok when distrusting
            # more points: We don't have to check the total length like in the case
            # below.

            coords = coords[trusted_sub_mask]
            cartesian_distances = snapping_dists[trusted_sub_mask]
            locations = locations[trusted_sub_mask]
            for i_trusted, i_global in enumerate(np.nonzero(trusted_mask)[0]):
                sli = slice(i_trusted, i_trusted + 1)
                ppoints = ProjectedPoints(
                    coords=coords[sli],
                    locations=locations[sli],
                    cartesian_distances=cartesian_distances[sli],
                )
                ppoints_list[i_global] = ppoints
        else:
            logger.debug("spacing of distances is not OK, untrusting all distances")
            # if the spacing was not ok, it can still be bad after untrusting all points
            # since the total length can be too short for the points
            untrusted_dists = np.full_like(self.dists, np.NaN)
            if not traj_spacing_ok(untrusted_dists):
                raise SnappingError(
                    f"trajectory of length {self.trajectory.length:g} is too short "
                    f"for {len(self)} points with "
                    f"min_spacing of {params.min_spacing:g}",
                )

        return ppoints_list

    def set_untrusted_candidates_inplace(
        self,
        params: SnappingParams,
        ppoints_list: list[typing.Optional[ProjectedPoints]],
    ) -> list[ProjectedPoints]:
        """Set untrusted snapping candidates inplace.

        Return the same list with altered type hint. Untrusted snapping candidates are
        generated by intersecting the trajectory with a ball around each untrusted point
        and discretizing the intersection. The radii and the discretization step are
        controlled by `params`.

        Existing (trusted) candidates are untouched.
        """
        mask = np.array([ppoints is None for ppoints in ppoints_list])
        lfs = self.line_fractions[mask]
        shortest_distances = self.projected_points.cartesian_distances[mask]
        square_radii = params.get_max_snapping_dists(shortest_distances)
        square_radii **= 2
        sliced_lfs_list = lfs.select_balls(square_radii)
        segment_lengths = np.diff(self.trajectory.dists)
        for i, sliced_lfs in zip(np.nonzero(mask)[0], sliced_lfs_list):
            if len(sliced_lfs) == 0:
                raise SnappingError(
                    f"untrusted point {i} is too far away from the trajectory",
                )
            ppoints = sliced_lfs.discretize(segment_lengths, params.sampling_step)
            assert len(ppoints), "non empty result expected for non empty sliced_lfs"
            ppoints_list[i] = ppoints
        assert all(ppoints is not None for ppoints in ppoints_list)
        return typing.cast(list[ProjectedPoints], ppoints_list)

    def snap_fallback(
        self, params: SnappingParams, reverse_order_allowed: bool
    ) -> list[ProjectedPoints]:
        distances_list = [
            np.linspace(self.trajectory.d_min, self.trajectory.d_max, len(self))
        ]
        if reverse_order_allowed:
            distances_list.append(distances_list[0][::-1])
        solutions: list[ProjectedPoints] = []
        for distances in distances_list:
            locations = self.trajectory.locate(distances)
            coords = interpolate(self.trajectory.xyz, locations)
            cartesian_distances = np.linalg.norm(coords - self.xyz, axis=1)
            max_snapping_distances = params.get_max_snapping_dists(
                self.projected_points.cartesian_distances
            )
            if np.all(cartesian_distances <= max_snapping_distances):
                ppoints = ProjectedPoints(coords, locations, cartesian_distances)
                solutions.append(ppoints)
        if not solutions:
            raise SnappingError(
                "All fallback solutions for at least one point are too far away."
            )

        n_solutions = len(solutions)
        result: list[ProjectedPoints] = []
        for i_point in range(len(self)):
            ppoints = ProjectedPoints.empty(n_points=n_solutions, n_cartesian=3)
            for i_solution in range(n_solutions):
                ppoints[i_solution : i_solution + 1] = solutions[i_solution][
                    i_point : i_point + 1
                ]
            result.append(ppoints)
        return result

    def route_candidates(
        self, params: SnappingParams, ppoints_list: list[ProjectedPoints]
    ) -> tuple[ProjectedPoints, float]:
        """Find best candidates by routing in a DAG.

        Return best candidates and total path cost (sum of square distances of best
        candidates to the trajectory).

        For each point we have candidate positions on the trajectory. Following the
        order of the points, we have a layered directed acyclic graph (DAG) when we
        connect all the candidates between each layer.

        If the minimum spacing is fine, the node penalty is the square cartesian
        distance of the original point to the candidate location. Otherwise it is
        infinite. Using those weights, we find the admissible solution that respects the
        min spacing and minimizes the sum of square distances to the trajectory (if it
        exists) since that will be the path length in the end.

        The weights of the edges between two layers are either infinite if the candidate
        pair violates the minimum spacing or depend on the target node only. This
        simplifies the shortest path calculation.

        The projected points for each input point have to be ordered by ascending
        position on the trajectory (not checked).
        """
        # implementation note: bisect does not support keys, thus we use separate lists
        # for each property instead of packing all properties into an object and using
        # one list

        # set up artificial start layer connected to all candidates of the real first
        # layer:

        reachable_candidates = [[0]]
        reachable_traj_distances = [[-float("inf")]]  # leftmost, thus always connected
        reachable_costs = [[0.0]]
        # index into the lists from the previous layer for every entry in the current
        # layer for shortest path reconstruction
        reachable_predecessors: list[list[typing.Optional[int]]] = [[None]]

        # artificial end layer connected to all reachable candidates of the real last
        # layer
        last_layer = len(ppoints_list) + 1

        for i_layer in range(1, last_layer + 1):
            if i_layer == last_layer:
                traj_dists = np.full(1, np.inf)  # rightmost, thus always connected
                n_points = 1
                # mypy cannot see that this is float by default
                candidate_costs = np.zeros(1, dtype=float)
            else:
                ppoints = ppoints_list[i_layer - 1]
                traj_dists = interpolate(self.trajectory.dists, ppoints.locations)
                n_points = len(ppoints)
                candidate_costs = ppoints.cartesian_distances**2
            prev_traj_dists = reachable_traj_distances[i_layer - 1]
            prev_costs = reachable_costs[i_layer - 1]
            lo = 0  # for speeding up bisection search taking order into account
            reachable_candidates.append([])
            reachable_traj_distances.append([])
            reachable_costs.append([])
            reachable_predecessors.append([])
            logger.debug("routing layer %d with %d points", i_layer, n_points)
            cum_min, cum_argmin = cumulative_min_and_argmin(prev_costs)
            for i_candidate in range(n_points):
                traj_dist = traj_dists[i_candidate]
                rightmost = (
                    bisect_right(prev_traj_dists, traj_dist - params.min_spacing, lo=lo)
                    - 1
                )
                lo = max(rightmost, 0)
                if rightmost >= 0:
                    best_prev_cost = cum_min[rightmost]
                    best_prev_i_reachable_cand = cum_argmin[rightmost]
                    reachable_candidates[i_layer].append(i_candidate)
                    reachable_traj_distances[i_layer].append(traj_dist)
                    reachable_costs[i_layer].append(
                        best_prev_cost + candidate_costs[i_candidate]
                    )
                    reachable_predecessors[i_layer].append(best_prev_i_reachable_cand)
            if not reachable_candidates[i_layer]:
                assert i_layer != last_layer, "last layer shall be connected"
                raise SnappingError(
                    "no solution possible for given max snapping distances "
                    "and min spacing (empty admissible candidates for point at "
                    f"index {i_layer - 1})"
                )

        path = self.reconstruct_path(
            ppoints_list, reachable_candidates, reachable_predecessors
        )
        total_cost = reachable_costs[-1][0]
        return path, total_cost

    def reconstruct_path(
        self,
        ppoints_list: list[ProjectedPoints],
        reachable_candidates: list[list[int]],
        reachable_predecessors: list[list[typing.Optional[int]]],
    ) -> ProjectedPoints:
        winner_indices: list[int] = [-1] * len(ppoints_list)
        predecessor = 0
        for i_layer in range(len(ppoints_list) + 1, 1, -1):
            next_predecessor = reachable_predecessors[i_layer][predecessor]
            assert next_predecessor is not None
            predecessor = next_predecessor
            winner_indices[i_layer - 2] = reachable_candidates[i_layer - 1][predecessor]
        ppoints = ProjectedPoints.empty(n_points=len(ppoints_list), n_cartesian=3)
        for path_index, winner_index in enumerate(winner_indices):
            ppoints[path_index : path_index + 1] = ppoints_list[path_index][
                winner_index : winner_index + 1
            ]
        return ppoints


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
