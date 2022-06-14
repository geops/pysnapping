import typing
from functools import partial
import logging

import pyproj
import numpy as np
from numpy.typing import ArrayLike

from .ordering import order_ok
from .linear_referencing import (
    Location,
    substring,
    find_location,
    interpolate,
    project,
)
from .util import iter_consecutive_groups
from . import SnappingError


logger = logging.getLogger(__name__)

WGS84_GEOD = pyproj.Geod(ellps="WGS84")
WGS84_CRS = pyproj.CRS.from_epsg(4326)


def array_chk(
    data: ArrayLike,
    shape_template: typing.Tuple[
        typing.Union[
            None, int, typing.Tuple[typing.Optional[int], typing.Optional[int]]
        ],
        ...,
    ],
    chk_finite: bool = False,
    **kwargs,
) -> np.ndarray:
    arr = np.array(data, **kwargs)
    temp_axes = len(shape_template)
    arr_axes = len(arr.shape)
    if arr_axes != temp_axes:
        raise ValueError(f"wrong number of axes (expected {temp_axes}, got {arr_axes})")
    for i, (n, n_expected) in enumerate(zip(arr.shape, shape_template)):
        if n_expected is None:
            pass
        elif isinstance(n_expected, int):
            if n != n_expected:
                raise ValueError(
                    f"axis {i} has wrong length (expected {n_expected}, got {n})"
                )
        else:
            n_min, n_max = n_expected
            if n_min is not None:
                if n < n_min:
                    raise ValueError(
                        f"axis {i} has wrong length (expected at least {n_expected}, got {n})"
                    )
            elif n_max is not None:
                if n > n_max:
                    raise ValueError(
                        f"axis {i} has wrong length (expected at most {n_expected}, got {n})"
                    )
    if chk_finite and not np.all(np.isfinite(arr)):
        raise ValueError("input not finite")
    return arr


class Trajectory:
    """A line string with optional travel distances for its vertices.

    The distances are to be interpreted as travel distances along
    the LineString in an arbitrary unit with arbitrary offset.

    Distances are allowed to contain NaNs and they are allowed to decrease.

    The interpretation of the coordinates is given by `crs`.
    """

    __slots__ = ("crs", "xyd")

    crs: pyproj.CRS
    xyd: np.ndarray

    def __init__(self, xyd: ArrayLike, crs: pyproj.CRS = WGS84_CRS):
        self.crs = crs
        self.xyd = array_chk(xyd, ((2, None), 3), dtype=float)
        if not np.all(np.isfinite(self.xyd[:, :2])):
            raise ValueError("coords have to be finite")

    def __len__(self) -> int:
        return len(self.xyd)

    @property
    def xy(self):
        return self.xyd[:, :2]

    @property
    def x(self):
        return self.xyd[:, 0]

    @property
    def y(self):
        return self.xyd[:, 1]

    @property
    def dists(self):
        return self.xyd[:, 2]


class TrajectoryTrip:
    """A sequence of points along a trajectory with optional distance and time information.

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

    trajectory: Trajectory
    xydt: np.ndarray

    def __init__(self, trajectory: Trajectory, xydt: ArrayLike):
        self.trajectory = trajectory
        self.xydt = array_chk(xydt, (None, 4), dtype=float)
        if not np.all(np.isfinite(self.xydt[:, :2])):
            raise ValueError("coords have to be finite")

    def __len__(self) -> int:
        return len(self.xydt)

    @property
    def dists(self):
        return self.xydt[:, 2]

    @property
    def xy(self):
        return self.xydt[:, :2]

    @property
    def x(self):
        return self.xyd[:, 0]

    @property
    def y(self):
        return self.xyd[:, 1]

    @property
    def times(self):
        return self.xydt[:, 3]

    def get_reverse_order_allowed(self) -> bool:
        return bool(np.all(np.isnan(self.dists)))


class WGS84Trajectory:
    """A line string on the WGS84 (EPSG:4326) ellipsoid with geodesic travel distances.

    Distances are the cumulative geodesic distances along the trajectory in meters.
    Distances do not have to start at 0. An arbitrary offset is allowed. This is useful
    for relating parts of a trajectory to the original trajectory.
    """

    __slots__ = ("lat_lon_d",)

    lat_lon_d: np.ndarray

    def __init__(self, lat_lon: ArrayLike):
        lat_lon = array_chk(
            lat_lon, ((2, None), 2), chk_finite=True, dtype=float, copy=False
        )
        self.lat_lon_d = np.empty((len(lat_lon), 3))
        self.lat_lon[...] = lat_lon

        self.dists[0] = 0
        segment_lengths = WGS84_GEOD.inv(
            lons1=self.lon[:-1],
            lats1=self.lat[:-1],
            lons2=self.lon[1:],
            lats2=self.lat[1:],
        )[2]
        np.cumsum(segment_lengths, out=self.dists[1:])

    def __len__(self) -> int:
        return len(self.lat_lon_d)

    @property
    def d_min(self) -> float:
        return float(self.lat_lon_d[0, 2])

    @property
    def d_max(self) -> float:
        return float(self.lat_lon_d[-1, 2])

    @property
    def lat_lon(self):
        return self.lat_lon_d[:, :2]

    @property
    def lat(self):
        return self.lat_lon_d[:, 0]

    @property
    def lon(self):
        return self.lat_lon_d[:, 1]

    @property
    def dists(self):
        return self.lat_lon_d[:, 2]

    @property
    def length(self):
        return self.lat_lon_d[-1, 2] - self.lat_lon_d[0, 2]

    @classmethod
    def from_trusted_data(cls, lat_lon_d: ArrayLike, copy: bool = True):
        """Create a WGS84Trajectory from existing coordinates and distances.

        No checks are performed.
        """
        trajectory = cls.__new__(cls)
        trajectory.lat_lon_d = np.array(lat_lon_d, dtype=float, copy=copy)

    def copy(self):
        return self.__class__.from_trusted_data(self.lat_lon_d)

    def split(
        self, where: typing.Sequence[typing.Union[Location, "WGS84SnappedTripPoint"]]
    ):
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
                substring(self.lat_lon_d, locations[i], locations[i + 1])
            )
            for i in range(len(locations) - 1)
        ]

    def locate(self, dist: float, **kwargs) -> Location:
        return find_location(self.dists, dist, **kwargs)


class WGS84TrajectoryTrip:
    """A sequence of points along a WGS84Trajectory with mandatory distance information.

    The distances are the geodesic travel distances along the trajectory in meters.
    All distances have to be finite, inside the distance range of the reference trajectory,
    and increasing in steps of at least `min_spacing` (in meters like distances).

    `dists_trusted` is a Boolean array indicating whether the corresponding distance is accurate or
    merely a rough hint.

    `reverse_order_allowed` tells whether the points could also be in reverse order.
    This flag has to be `False` if any distance is trusted.

    `atol` is the absolute tolerance (in meters) when checking validity of distances.
    """

    __slots__ = (
        "trajectory",
        "lat_lon_d",
        "dists_trusted",
        "min_spacing",
        "reverse_order_allowed",
        "atol",
    )

    trajectory: WGS84Trajectory
    lat_lon_d: np.ndarray
    dists_trusted: np.ndarray
    min_spacing: float
    reverse_order_allowed: bool
    atol: float

    def __init__(
        self,
        trajectory: WGS84Trajectory,
        lat_lon_d: ArrayLike,
        dists_trusted: ArrayLike,
        min_spacing: float,
        reverse_order_allowed: bool,
        atol: float = 5.0,
    ):
        self.trajectory = trajectory
        self.lat_lon_d = array_chk(lat_lon_d, (None, 3), chk_finite=True, dtype=float)
        self.dists_trusted = array_chk(
            dists_trusted, (len(self.lat_lon_d),), dtype=bool
        )

        self.min_spacing = min_spacing
        self.reverse_order_allowed = reverse_order_allowed
        self.atol = atol
        if self.reverse_order_allowed and np.any(self.dists_trusted):
            raise ValueError("reverse order is allowed but there are trusted dists")
        if not self.order_ok(self.lat_lon_d[:, 2]):
            raise ValueError("bad distances")

    def __len__(self) -> int:
        return len(self.lat_lon_d)

    @property
    def lat_lon(self):
        return self.lat_lon_d[:, :2]

    @property
    def lat(self):
        return self.lat_lon_d[:, 0]

    @property
    def lon(self):
        return self.lat_lon_d[:, 1]

    @property
    def dists(self):
        return self.lat_lon_d[:, 2]

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
        if len(trusted_indices) > 0 and self.reverse_order_allowed:
            raise ValueError(
                "reverse order allowed not possible with trusted distances"
            )
        is_reversed = False

        # trusted distances
        for i in trusted_indices:
            snapped_points[i] = WGS84SnappedTripPoint(self, i, self.dists[i])

        # every group of consecutive points with untrusted distances is an isolated problem
        untrusted_indices = np.nonzero(np.logical_not(self.dists_trusted))[0]
        for group_indices in iter_consecutive_groups(untrusted_indices):
            # if is_reversed is True, the loop has exactly one pass, so we can directly assign
            snapped_points[group_indices], is_reversed = self._snap_untrusted(
                indices=group_indices,
                convergence_accuracy=convergence_accuracy,
                n_iter_max=n_iter_max,
            )

        return WGS84SnappedTripPoints(snapped_points.tolist(), is_reversed)

    def order_ok(
        self,
        dists_or_snapped_points: typing.Iterable[
            typing.Union["WGS84SnappedTripPoint", float]
        ],
        d_min: typing.Optional[float] = None,
        d_max: typing.Optional[float] = None,
    ) -> bool:
        if d_min is None:
            d_min = self.trajectory.d_min
        if d_max is None:
            d_max = self.trajectory.d_max
        return order_ok(
            [
                item.trajectory_distance
                if isinstance(item, WGS84SnappedTripPoint)
                else item
                for item in dists_or_snapped_points
            ],
            v_min=d_min,
            v_max=d_max,
            d_min=self.min_spacing,
            atol=self.atol,
        )

    def _snap_untrusted(
        self,
        indices: typing.List[int],
        convergence_accuracy: float,
        n_iter_max: int,
    ) -> typing.Tuple[typing.List["WGS84SnappedTripPoint"], bool]:
        # calculate free space where we can snap to without getting too close
        # to points with trusted dists
        left = indices[0] - 1
        right = indices[-1] + 1
        if left < 0:
            d_min = self.trajectory.d_min
        else:
            d_min = self.dists[left] + self.min_spacing
        if right >= len(self):
            d_max = self.trajectory.d_max
        else:
            d_max = self.dists[right] - self.min_spacing
        n_points = len(indices)
        available_length = d_max - d_min
        required_length = (n_points - 1) * self.min_spacing
        if required_length > available_length:
            raise SnappingError("not enough space")

        # the linestring we can snap to without violating d_min / d_max
        target_lat_lon_d = substring(
            self.trajectory.lat_lon_d,
            self.trajectory.locate(d_min),
            self.trajectory.locate(d_max),
        )

        # first try just projecting; if this works, we're done
        snapped_points = [
            WGS84SnappedTripPoint(
                self,
                i,
                float(
                    interpolate(
                        target_lat_lon_d[:, 2],
                        project(target_lat_lon_d[:, :2], self.lat_lon[i]).location,
                    )
                ),
            )
            for i in indices
        ]

        d_ok = partial(self.order_ok, d_min=d_min, d_max=d_max)
        if d_ok(snapped_points):
            is_reversed = False
        elif self.reverse_order_allowed and d_ok(reversed(snapped_points)):
            is_reversed = True
        else:
            snapped_points = self._snap_untrusted_iteratively(
                indices=indices,
                d_min=d_min,
                d_max=d_max,
                reverse=False,
                convergence_accuracy=convergence_accuracy,
                n_iter_max=n_iter_max,
            )
            is_reversed = False
            if self.reverse_order_allowed:
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
                    # we reverse later after order_ok check
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
        if reverse:
            indices = indices[::-1]
            point_dists = self.dists[indices]
            point_dists = (d_min + d_max) - point_dists
        else:
            point_dists = self.dists[indices]

        if not self.order_ok(point_dists, d_min=d_min, d_max=d_max):
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
            region_boundaries[1:-2:2] = mid_dists - 0.5 * self.min_spacing
            region_boundaries[2:-1:2] = mid_dists + 0.5 * self.min_spacing

            logger.debug(
                "n_iter=%d: delta=%.2e point_dists=%s region_boundaries=%s",
                n_iter,
                delta,
                point_dists,
                region_boundaries,
            )
            regions = [
                substring(
                    self.trajectory.lat_lon_d,
                    self.trajectory.locate(region_boundaries[i]),
                    self.trajectory.locate(region_boundaries[i + 1]),
                )
                for i in range(0, len(region_boundaries) - 1, 2)
            ]
            points_projected_to_regions = [
                project(region[:, :2], self.lat_lon[i])
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

        return [WGS84SnappedTripPoint(self, i, d) for i, d in zip(indices, point_dists)]


class WGS84SnappedTripPoint:
    __slots__ = (
        "trip",
        "index",
        "trajectory_distance",
    )

    trip: WGS84TrajectoryTrip
    index: int
    trajectory_distance: float

    def __init__(
        self,
        trip: WGS84TrajectoryTrip,
        index: int,
        trajectory_distance: float,
    ):
        self.trip = trip
        self.index = index
        self.trajectory_distance = trajectory_distance

    def get_trajectory_location(self) -> Location:
        return find_location(self.trip.trajectory.dists, self.trajectory_distance)

    def get_lat_lon(self) -> np.ndarray:
        return interpolate(self.trip.trajectory.lat_lon, self.get_trajectory_location())

    def get_geodesic_snapping_distance(self) -> float:
        """Get the geodesic distance between source and snapped point in meters"""
        source_lat_lon = self.trip.lat_lon[self.index]
        snapped_lat_lon = self.get_lat_lon()
        return float(
            WGS84_GEOD.inv(
                lats1=source_lat_lon[0],
                lons1=source_lat_lon[1],
                lats2=snapped_lat_lon[0],
                lons2=snapped_lat_lon[1],
            )[2]
        )

    def snapping_distance_valid(self, rtol: float = 3.0, atol: float = 200.0) -> bool:
        """Determine whether the geodesic snapping distance is valid.

        The geodesic distance between the source and snapped point is compared to the
        (approximate) shortest geodesic distance between the source point and the trajectory.
        The geodesic distance to the trajectory is approximated by the geodesic distance between
        the source point and its eucledian projection onto the trajectory.

        rtol/atol -- Relative/absolute tolerance.
                     Default means that a snapped point may be 3 times as far away from the
                     original point than the closest point on the trajectory plus 200 meters.
        """
        source_lat_lon = self.trip.lat_lon[self.index]
        projected_lat_lon = project(self.trip.trajectory.lat_lon, source_lat_lon).coords
        projected_distance = WGS84_GEOD.inv(
            lats1=source_lat_lon[0],
            lons1=source_lat_lon[1],
            lats2=projected_lat_lon[0],
            lons2=projected_lat_lon[1],
        )[2]
        snapping_distance = self.get_geodesic_snapping_distance()
        diff = abs(snapping_distance - projected_distance)
        return diff <= rtol * projected_distance + atol


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

    def snapping_distances_valid(self, rtol: float = 3.0, atol: float = 200.0):
        """Determine whether all geodesic snapping distances are valid.

        See WGS84SnappedTripPoint.snapping_distance_valid for meaning of parameters.
        """
        return all(
            p.snapping_distance_valid(rtol=rtol, atol=atol) for p in self.snapped_points
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
        in travel direction, use `get_inter_point_linestring_lat_lon_in_travel_direction` instead.
        """
        if len(self.snapped_points) < 2:
            raise ValueError("at least two snapped points are required")

        trajectory = self.snapped_points[0].trip.trajectory

        if self.dists_descending:
            return trajectory.split(self.snapped_points[::-1])[::-1]
        else:
            return trajectory.split(self.snapped_points)

    def get_inter_point_linestrings_in_travel_direction(
        self,
    ) -> typing.List[np.ndarray]:
        """Split the trajectory at the snapped points.

        Return linestring lat,lon coords between consecutive pairs of trip points
        in the same order as in the trip.
        All linestrings are oriented in travel direction of the trip.
        """
        trajectories = self.get_inter_point_trajectories()

        # t.lat_lon is a view into t.lat_lon_d, so we copy to free distance data
        if self.dists_descending:
            return [t.lat_lon[::-1].copy() for t in trajectories]
        else:
            return [t.lat_lon.copy() for t in trajectories]
