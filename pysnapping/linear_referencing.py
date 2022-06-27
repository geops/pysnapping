"""Linear referencing for arbitrary dimensions.

An alternative to the very limited shapely linear referencing tools.
"""

import typing

import numpy as np
from numpy.core.umath import clip  # type: ignore


class Location(typing.NamedTuple):
    """A location along a linear feature"""

    segment: int
    fraction: float

    def clip(self):
        if 0 <= self.fraction <= 1:
            # NamedTuple is immutable, so avoid making a copy if no clipping is needed
            return self
        else:
            return self.__class__(
                segment=self.segment, fraction=min(max(self.fraction, 0.0), 1.0)
            )


# TODO (nice to have / performance):
# * support snapping multiple points at once with another numpy axis
# * support snapping to substring without actually making the substring
# * make infinite_head/tail parameters for project (don't fix on the instance)
class ProjectionTarget:
    """An N-dimensional linestring optimized as a projection target."""

    __slots__ = (
        "segment_starts",
        "segment_dirs",
        "norm_segment_dirs_squared",
        "short_segments",
        "infinite_head",
        "infinite_tail",
    )

    segment_starts: np.ndarray
    segment_dirs: np.ndarray
    norm_segment_dirs_squared: np.ndarray
    short_segments: np.ndarray
    infinite_head: bool
    initite_tail: bool

    def __init__(
        self,
        line_string_coords: np.ndarray,
        infinite_head: bool = False,
        infinite_tail: bool = False,
        epsilon: float = 1e-10,
    ):
        """Initialize ProjectionTarget.

        `line_string_coords` is a sequence of coords (vertex index at axis 0).
        `infinite_head` and `infinite_tail` control whether
        first/last segment should be extended to infinity.
        If enabled, the fraction of the location might be < 0 or > 1 after projecting.
        `epsilon` is used to identify very short segments (for numerical reasons).

        Attention: A mixture of views and pre-processed data of `line_string_coords` is
        stored for performance reasons. Modifying `line_string_coords` leads to undefined
        behavior when projecting later! If you need to modify, pass a copy to the constructor.
        """
        # ndim is the number of axes, not the number of cartesian dimensions
        if line_string_coords.ndim != 2:
            raise ValueError("line_string_coords has to be a sequence of points")
        n_vertices = line_string_coords.shape[0]
        if n_vertices < 2:
            raise ValueError("at least two vertices are needed")

        self.segment_starts = line_string_coords[:-1]
        segment_ends = line_string_coords[1:]
        self.segment_dirs = segment_ends - self.segment_starts

        # sum(-1) means sum over last axis
        self.norm_segment_dirs_squared = (self.segment_dirs**2).sum(-1)
        short_segment_mask = self.norm_segment_dirs_squared < epsilon**2
        self.infinite_head = infinite_head
        self.infinite_tail = infinite_tail
        if infinite_head and short_segment_mask[0]:
            raise ValueError("head segment too short, consider using simplify first")
        if infinite_tail and short_segment_mask[-1]:
            raise ValueError("tail segment too short, consider using simplify first")
        # indexing with integers is faster than indexing with a mask for the
        # common use case of a sparse mask
        self.short_segments = np.nonzero(short_segment_mask)[0]
        self.norm_segment_dirs_squared[self.short_segments] = 1.0

    def project(self, point_coords: np.ndarray) -> "ProjectedPoint":
        """Project a point to the linestring.

        This is basically a generalization and optimization of shapely.geometry.LineString.project.

        `point_coords` are the coordinates of the point.
        """
        if point_coords.ndim != 1:
            raise ValueError("point has to contain exactly 1 axis")
        if point_coords.shape != self.segment_starts.shape[1:]:
            raise ValueError("point and target cartesian dimensions do not match")

        point_dirs = point_coords[None, :] - self.segment_starts

        # many scalar products
        fractions = (self.segment_dirs * point_dirs).sum(-1)

        fractions /= self.norm_segment_dirs_squared
        fractions[self.short_segments] = 0.5

        if self.infinite_head or self.infinite_tail:
            clip_slice = slice(
                1 if self.infinite_head else None, -1 if self.infinite_tail else None
            )
            clip(fractions[clip_slice], 0.0, 1.0, out=fractions[clip_slice])
            if self.infinite_head and (len(fractions) > 1 or not self.infinite_tail):
                fractions[0] = min(fractions[0], 1)
            if self.infinite_tail and (len(fractions) > 1 or not self.infinite_head):
                fractions[-1] = max(fractions[-1], 0)
        else:
            # different case just for speedup
            clip(fractions, 0.0, 1.0, out=fractions)

        projected_points = fractions[:, None] * self.segment_dirs
        projected_points += self.segment_starts
        # ranks are squared cartesian distances (sqrt is monotone, so we get the same minimum
        # as for cartesian distances with less computational effort)
        tmp = point_coords[None, :] - projected_points
        tmp *= tmp
        ranks = tmp.sum(-1)
        segment = int(np.argmin(ranks))

        return ProjectedPoint(
            projected_points[segment].copy(),
            Location(segment, float(fractions[segment])),
            float(ranks[segment] ** 0.5),
        )


class ProjectedPoint(typing.NamedTuple):
    coords: np.ndarray
    location: Location
    cartesian_distance: float


def check_n_segments(data: np.ndarray, axis: int = 0) -> None:
    n_segments = data.shape[axis] - 1
    if n_segments <= 0:
        raise ValueError(f"Axis {axis} is too short. At least 2 entries needed.")


def interpolate(
    data: np.ndarray, location: Location, axis: int = 0, extrapolate: bool = False
) -> np.ndarray:
    """Interpolate data interpreted as a tensor-valued linestring.

    This is a generalization of shapely.geometry.LineString.interpolate.

    `axis` is the vertex index axis.
    """
    check_n_segments(data, axis)
    if not extrapolate:
        location = location.clip()

    if axis != 0:
        # a view of data with vertex index axis leftmost for convenience
        data = np.moveaxis(data, axis, 0)

    f = location.fraction
    return (1.0 - f) * data[location.segment] + f * data[location.segment + 1]


def substring(
    data: np.ndarray,
    start: Location,
    end: Location,
    axis: int = 0,
) -> np.ndarray:
    """Extract a substring copy of data interpeted as a tensor-valued linestring.

    This is a generalization of shapely.ops.substring.
    In contrast to shapely, two identical points are returned for a zero-length substring
    and the selection of the substring extent is different.
    Start and end fractions are clipped to [0, 1] (extrapolation is not implemented yet).
    If end lies before start, a zero length substring from start to start is returned.
    Start/end points are interpolated linearly.

    `axis` is the vertex index axis.
    """
    check_n_segments(data, axis)
    start = start.clip()
    end = end.clip()
    end = max(start, end)

    if start.fraction == 0.0:
        interpolate_start = False
        first = start.segment
    elif start.fraction == 1.0:
        interpolate_start = False
        first = start.segment + 1
    else:
        interpolate_start = True
        first = start.segment + 1

    if end.fraction == 0.0:
        interpolate_end = False
        last = end.segment
    elif end.fraction == 1.0:
        interpolate_end = False
        last = end.segment + 1
    else:
        interpolate_end = True
        last = end.segment

    sub_shape = list(data.shape)
    n_points = last - first + 1 + interpolate_start + interpolate_end
    assert n_points >= 1
    sub_shape[axis] = max(n_points, 2)
    sub_data = np.empty(tuple(sub_shape))

    # views have `axis` leftmost for convenient indexing
    sub_data_view = np.moveaxis(sub_data, axis, 0)
    data_view = np.moveaxis(data, axis, 0)

    sub_data_view[int(interpolate_start) : n_points - int(interpolate_end)] = data_view[
        first : last + 1
    ]
    if interpolate_start:
        sub_data_view[0] = interpolate(data_view, start)
    if interpolate_end:
        sub_data_view[-1] = interpolate(data_view, end)
    if n_points == 1:
        sub_data_view[1] = sub_data_view[0]
    return sub_data


def locate(distances: np.ndarray, distance: float) -> Location:
    """Find location of a distance in a sequence of distances.

    `distances` has to be sorted in ascending order (not checked).

    Location is not extrapolated for distance outside bounds.
    (maybe we add this feature later optionally)

    If there are repeated consecutive distances, the location
    will be in the middle.

    Location(i, 0.0) is preferred over Location(i - 1, 1.0)
    as long as i is a valid segment index.
    """
    check_n_segments(distances)

    # find i1 and i2 such that distances[i1] < distance < distances[i2]
    # with distances[-1] (not in the python sense) := -infinity
    # and distances[len(distances)] := +infinity
    i1 = int(np.searchsorted(distances, distance, side="left")) - 1
    i2 = int(np.searchsorted(distances, distance, side="right"))

    if i1 + 1 == i2:
        # We are on a segment with non-zero length or out of bounds.
        # Handle out of bounds and interpolate else.
        if i1 == -1:
            return Location(0, 0.0)
        elif i2 == len(distances):
            return Location(i2 - 2, 1.0)
        else:
            return Location(
                i1, (distance - distances[i1]) / (distances[i2] - distances[i1])
            )
    else:
        # We exactly hit a distance from distances.
        # Take the middle segment of all those distances.
        # If there is an even number of such segments, bias to the right with fraction 0.0.
        # If we overflow to the right in the even case, back off by one segment and use fraction
        # 1.0 instead of 0.0.
        # If there is an odd number of segments, take the middle one with fraction 0.5.
        segment, mod = divmod(i1 + i2, 2)
        if segment == len(distances) - 1:
            # segment does not exist, back off to the left
            return Location(segment - 1, 1.0)
        else:
            return Location(segment, 0.5 * mod)
