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
    assert n_points >= 2
    sub_shape[axis] = n_points
    sub_data = np.empty(tuple(sub_shape))

    # views have `axis` leftmost for convenient indexing
    sub_data_view = np.moveaxis(sub_data, axis, 0)
    data_view = np.moveaxis(data, axis, 0)

    sub_data_view[int(interpolate_start) : n_points - interpolate_end] = data_view[
        first : last + 1
    ]
    if interpolate_start:
        sub_data_view[0] = interpolate(data_view, start)
    if interpolate_end:
        sub_data_view[-1] = interpolate(data_view, end)
    return sub_data


def project(
    line_string_coords: np.ndarray,
    point_coords: np.ndarray,
    infinite_head: bool = False,
    infinite_tail: bool = False,
    epsilon: float = 1e-10,
) -> ProjectedPoint:
    """Project point to linestring in N dimensions.

    This is a generalization of shapely.geometry.LineString.project.

    `line_string_coords` is a sequence of coords (vertex index at axis 0).
    `point_coords` are the coordinates of the point.
    `infinite_head` and `infinite_tail` control whether
    first/last segment should be extended to infinity.
    If enabled, the fraction of the location might be < 0 or > 1
    `epsilon` is used to identify very short segments.
    """
    # ndim is the number of axes, not the number of cartesian dimensions
    if line_string_coords.ndim != 2:
        raise ValueError("line_string_coords has to be a sequence of points")
    if point_coords.ndim != 1:
        raise ValueError("point has to contain exactly 1 axis")
    n_vertices = line_string_coords.shape[0]
    if n_vertices < 2:
        raise ValueError("at least two vertices are needed")
    if point_coords.shape != line_string_coords.shape[1:]:
        raise ValueError("point and vertex dimensions do not match")

    segment_starts = line_string_coords[:-1]
    segment_ends = line_string_coords[1:]
    segment_dirs = segment_ends - segment_starts

    # sum(-1) means sum over last axis
    norm_segment_dirs_squared = (segment_dirs**2).sum(-1)
    mask = norm_segment_dirs_squared < epsilon**2
    if infinite_head and mask[0]:
        raise ValueError("head segment too short, consider using simplify first")
    if infinite_tail and mask[-1]:
        raise ValueError("tail segment too short, consider using simplify first")
    norm_segment_dirs_squared[mask] = 1.0

    point_dirs = point_coords[None, :] - segment_starts

    # many scalar products
    fractions = (segment_dirs * point_dirs).sum(-1)

    fractions /= norm_segment_dirs_squared
    fractions[mask] = 0.5

    clip_slice = slice(1 if infinite_head else None, -1 if infinite_tail else None)
    clip(fractions[clip_slice], 0.0, 1.0, out=fractions[clip_slice])
    if infinite_head and (len(fractions) > 1 or not infinite_tail):
        fractions[0] = min(fractions[0], 1)
    if infinite_tail and (len(fractions) > 1 or not infinite_head):
        fractions[-1] = max(fractions[-1], 0)

    projected_points = segment_starts + fractions[:, None] * segment_dirs
    # ranks are squared cartesian distances (sqrt is monotone, so we get the same minimum
    # as for cartesian distances with less computational effort)
    ranks = ((point_coords[None, :] - projected_points) ** 2).sum(-1)
    segment = int(np.argmin(ranks))

    return ProjectedPoint(
        projected_points[segment].copy(),
        Location(segment, float(fractions[segment])),
        float(ranks[segment] ** 0.5),
    )


def find_location(
    distances: np.ndarray, distance: float, epsilon: float = 1e-10
) -> Location:
    """Find location of a distance in a sequence of distances.

    `distances` has to be sorted in ascending order (not checked).
    `distance` is clipped to [`distances[0]`, `distances[-1]`]
    `epsilon` is used to detect very short segments.
    """
    check_n_segments(distances)
    distance = min(max(distance, distances[0]), distances[-1])
    if distance == distances[0]:
        return Location(0, 0.0)
    elif distance == distances[-1]:
        return Location(len(distances) - 2, 1.0)
    else:
        # i2 is the smallest index with distance < distances[i2]
        i2 = int(np.searchsorted(distances, distance, side="right"))
        i1 = i2 - 1
        # since we treated the edge cases separately above, we know that
        # distances[i1] < distance < distances[i2] and i1 and i2 are inside bounds
        segment_length = distances[i2] - distances[i1]
        if segment_length > epsilon:
            fraction = (distance - distances[i1]) / segment_length
        else:
            fraction = 0.5
        # not sure if we can end up outside of [0, 1] due to floating point errors, so better clip
        return Location(i1, fraction).clip()
