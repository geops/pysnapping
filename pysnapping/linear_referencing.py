"""Linear referencing for arbitrary dimensions.

An alternative to the very limited shapely linear referencing tools.
Also suitable as an alternative to scipy.interp.interp1d when dealing with repeated x values.

The vertex index axis is always expected at axis 0.
Consider using np.moveaxis if your data is shaped differently.

This module is rather low-level without sanity checks and conversion of array-like to array.
"""

import typing
from math import floor

import numpy as np

# faster than np.clip
from numpy.core.umath import clip  # type: ignore

from . import ExtrapolationError


class Locations:
    """Multiple locations along a linear feature.

    Each location is described by two vertex indices and a fraction.
    Fraction 0 means at the first vertex. Fraction 1 means at the second vertex.
    Fraction f with 0 < f < 1 means between vertex1 and vertex2 (linear interpolation).
    Fraction f with f < 0 or f > 1 describes linear extrapolation.

    Usually to_vertices is from_vertices + 1 but not necessarily.
    A larger step is e.g. useful if you want to extrapolate with repeated x values,
    then you can skip segments until the "virtual segment" has a non-zero length.
    """

    __slots__ = ("from_vertices", "to_vertices", "fractions")

    from_vertices: np.ndarray
    to_vertices: np.ndarray
    fractions: np.ndarray

    def __init__(
        self, from_vertices: np.ndarray, to_vertices: np.ndarray, fractions: np.ndarray
    ):
        self.from_vertices = from_vertices
        self.to_vertices = to_vertices
        self.fractions = fractions

    def __getitem__(self, key):
        """Get a view or copy using only a subset of locations.

        View/copy behavior is like in numpy
        (copy for fancy indexing with integers or mask; view for slices).
        """
        if isinstance(key, (int, tuple)):
            raise KeyError(
                "indexing is only possible by a single slice, a single mask "
                "or a single index list/array"
            )
        return type(self)(
            from_vertices=self.from_vertices[key],
            to_vertices=self.to_vertices[key],
            fractions=self.fractions[key],
        )

    def __setitem__(self, key, item):
        self.from_vertices[key] = item.from_vertices
        self.to_vertices[key] = item.to_vertices
        self.fractions[key] = item.fractions

    def __len__(self) -> int:
        return len(self.from_vertices)

    @classmethod
    def empty(cls, n: int):
        shape = (n,)
        return cls(
            np.empty(shape, dtype=int),
            np.empty(shape, dtype=int),
            np.empty(shape),
        )

    def get_fractional_indices(self):
        from_vertices = np.array(self.from_vertices, dtype=float)
        ifrac = np.array(self.to_vertices, dtype=float)
        ifrac -= from_vertices
        ifrac *= self.fractions
        ifrac += from_vertices
        return ifrac

    def max(self, other):
        self_wins = self.get_fractional_indices() >= other.get_fractional_indices()
        return type(self)(
            np.where(self_wins, self.from_vertices, other.from_vertices),
            np.where(self_wins, self.to_vertices, other.to_vertices),
            np.where(self_wins, self.fractions, other.fractions),
        )


def location_to_single_segment(
    from_vertex: int,
    to_vertex: int,
    fraction: float,
    prefer_zero_fraction: bool,
    last_segment: int,
) -> typing.Tuple[int, float]:
    n_segments = to_vertex - from_vertex
    if n_segments == 1:
        segment = from_vertex
    elif not 0 <= fraction <= 1:
        raise ExtrapolationError(
            "cannot convert location extrapolating over multiple segments to single segment"
        )
    elif n_segments == 0:
        segment, fraction = from_vertex, 0.0
    else:
        pos = from_vertex + fraction * n_segments
        segment = floor(pos)
        # clip away possible floating point errors to stay inside [0, 1]
        fraction = min(max(pos - segment, 0.0), 1.0)

    if segment == last_segment + 1 and fraction == 0.0:
        segment = last_segment
        fraction = 1.0

    if prefer_zero_fraction and segment != last_segment and fraction == 1.0:
        segment += 1
        fraction = 0.0
    if not prefer_zero_fraction and segment != 0 and fraction == 0.0:
        segment -= 1
        fraction = 1.0
    assert 0 <= segment <= last_segment
    return segment, fraction


class ProjectedPoints:
    __slots__ = ("coords", "locations", "cartesian_distances")

    coords: np.ndarray
    locations: Locations
    cartesian_distances: np.ndarray

    def __init__(
        self, coords: np.ndarray, locations: Locations, cartesian_distances: np.ndarray
    ):
        self.coords = coords
        self.locations = locations
        self.cartesian_distances = cartesian_distances

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, key):
        """Get a view or copy using only a subset of the points.

        View/copy behavior is like in numpy
        (copy for fancy indexing with integers or mask; view for slices).
        """
        if isinstance(key, (int, tuple)):
            raise KeyError(
                "indexing is only possible by a single slice, a single mask "
                "or a single index list/array"
            )
        return type(self)(
            coords=self.coords[key],
            locations=self.locations[key],
            cartesian_distances=self.cartesian_distances[key],
        )

    def __setitem__(self, key, item):
        self.coords[key] = item.coords
        self.locations[key] = item.locations
        self.cartesian_distances[key] = item.cartesian_distances

    @classmethod
    def empty(cls, n_points: int, n_cartesian: int):
        return cls(
            np.empty((n_points, n_cartesian)),
            Locations.empty(n_points),
            np.empty((n_points,)),
        )


class ProjectionTarget:
    """An N-dimensional linestring optimized as a projection target."""

    __slots__ = (
        "segment_starts",
        "segment_dirs",
        "inv_norm_segment_dirs_squared",
        "short_segment_mask",
        "short_segments",
    )

    segment_starts: np.ndarray
    segment_dirs: np.ndarray
    inv_norm_segment_dirs_squared: np.ndarray
    short_segment_mask: np.ndarray
    short_segments: np.ndarray

    def __init__(
        self,
        line_string_coords: np.ndarray,
        epsilon: float = 1e-10,
    ):
        """Initialize ProjectionTarget.

        `line_string_coords` is a sequence of coords (vertex index at axis 0).
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
        self.inv_norm_segment_dirs_squared = (self.segment_dirs**2).sum(-1)
        self.short_segment_mask = self.inv_norm_segment_dirs_squared < epsilon**2
        self.short_segments = np.nonzero(self.short_segment_mask)[0]
        # avoid division by zero (will be overwritten later)
        self.inv_norm_segment_dirs_squared[self.short_segments] = 1.0
        # in general, multiplication is faster than division,
        # so we calculate the multiplicative inverse once
        self.inv_norm_segment_dirs_squared **= -1

    def get_line_fractions(self, point_coords: np.ndarray) -> "LineFractions":
        """Project points to infinite lines passing through linestring segments.

        Short segments always get fraction 0.5 (see `epsilon` parameter in constructor).
        """
        # many scalar products between segment directions and
        # point directions to get the fractions for each point/segment combination
        tmp = point_coords[:, np.newaxis, :] - self.segment_starts[np.newaxis, :, :]
        tmp *= self.segment_dirs
        fractions = tmp.sum(-1)
        fractions *= self.inv_norm_segment_dirs_squared[np.newaxis, :]
        # self.inv_norm_segment_dirs_squared is arbitrarily set to 1 for short segments.
        # If a segment is too short for a well defined orientation,
        # we always assume we are in the middle as stated in the docstring.
        fractions[:, self.short_segments] = 0.5

        return LineFractions(self, point_coords, fractions)

    def project(
        self,
        point_coords: np.ndarray,
        **kwargs,
    ) -> ProjectedPoints:
        """Project points to the linestring.

        This is a convenience method that chains `get_line_fractions` and
        `LineFractions.project`.

        Note that projecting multiple times with different `kwargs` is faster when calling
        `get_line_fractions` only once and then calling `LineFractions.project` multiple
        times.

        This is basically a generalization and optimization of shapely.geometry.LineString.project.

        `point_coords` are the coordinates of the points with vertex index axis at 0.
        `kwargs` are passed to `LineFractions.project`.
        """
        return self.get_line_fractions(point_coords).project(**kwargs)


class LineFractions:
    """Fractions of points projected to infinite lines running through each segment of a target."""

    __slots__ = ("target", "point_coords", "fractions")

    target: ProjectionTarget
    point_coords: np.ndarray
    fractions: np.ndarray

    def __init__(
        self, target: ProjectionTarget, point_coords: np.ndarray, fractions: np.ndarray
    ):
        self.target = target
        self.point_coords = point_coords
        self.fractions = fractions

    def __getitem__(self, key):
        """Get a view or copy using only a subset of points.

        View/copy behavior is like in numpy
        (copy for fancy indexing with integers or mask; view for slices).
        `target` is not copied.
        """
        if isinstance(key, (int, tuple)):
            raise KeyError(
                "indexing is only possible by a single slice, a single mask "
                "or a single index list/array"
            )
        return type(self)(
            target=self.target,
            point_coords=self.point_coords[key],
            fractions=self.fractions[key],
        )

    def project(
        self,
        head_segment: int = 0,
        head_fraction: float = 0.0,
        tail_segment: int = -1,
        tail_fraction: float = 1.0,
        out: typing.Optional[ProjectedPoints] = None,
    ) -> ProjectedPoints:
        """Project points to the linestring or a substring thereof.

        Clip fractions to segments and get ProjectedPoints for best clipped fractions.

        For each point, the projected point with the shortest distance to the point wins.
        If there are multiple segments with the shortest distance, the segment
        with the lowest index wins.

        `head_segment/fraction` and `tail_segment/fraction` control the clipping at the start/end.
        Default corresponds to clipping to the linestring boundaries.
        Use a negative value for `head_fraction` to allow projecting to an elongated head.
        `float("-inf")` can be used to simulate an infinite head.
        You can also use a value greater than 0.0 to shorten the head
        (useful for projecting to substrings).
        `head_segment` is the index of the head segment. All segments before `head_segment`
        are ignored in the minimum search.

        Similar for `tail_segment/fraction`.

        If tail lies before head, the substring to project on will be point-like from head to head
        and all points will end up there.

        The head/tail segment cannot be short if elongating head/tail (simplify the linestring to
        ensure this).
        """
        if head_fraction < 0.0 and self.target.short_segment_mask[head_segment]:
            raise ExtrapolationError(
                "head segment too short to elongate; consider using simplify first"
            )
        if tail_fraction > 1.0 and self.target.short_segment_mask[tail_segment]:
            raise ExtrapolationError(
                "tail segment too short to elongate; consider using simplify first"
            )

        # get rid of negative numbers
        head_segment, tail_segment, _ = slice(head_segment, tail_segment).indices(
            self.fractions.shape[1]
        )

        tail_segment, tail_fraction = max(
            (tail_segment, tail_fraction), (head_segment, head_fraction)
        )

        # views for the relevant segments (no copies made here)
        # and we bring everything to all 3 axes
        segment_slice = slice(head_segment, tail_segment + 1)
        point_coords = self.point_coords[:, np.newaxis, :]
        fractions = self.fractions[:, segment_slice, np.newaxis]
        segment_dirs = self.target.segment_dirs[np.newaxis, segment_slice, :]
        segment_starts = self.target.segment_starts[np.newaxis, segment_slice, :]
        del self  # protection against using unsliced data

        low: typing.Union[float, np.ndarray]
        high: typing.Union[float, np.ndarray]
        # different cases only for speeding up the default
        if head_fraction != 0.0:
            low = np.zeros((1, fractions.shape[1], 1))
            low[0, 0, 0] = head_fraction
        else:
            low = 0.0
        if tail_fraction != 1.0:
            high = np.ones((1, fractions.shape[1], 1))
            high[0, -1, 0] = tail_fraction
        else:
            high = 1.0

        fractions = clip(fractions, low, high)

        projected_points = fractions * segment_dirs
        projected_points += segment_starts
        # ranks are squared cartesian distances (sqrt is monotone, so we get the same minimum
        # as for cartesian distances with less computational effort)
        tmp = point_coords - projected_points
        tmp *= tmp
        ranks = tmp.sum(2)
        # index of winning segment for each point
        segments = np.argmin(ranks, axis=1)

        # We need to explicitly select the points by index.
        # If we would use `:` for the point axis, we would get a cartesian product
        # which is not what we want. We want a specific segment for a specific point each.
        point_indices = np.arange(len(point_coords))

        # take sqrt only for winners to get cartesian distances
        distances = ranks[point_indices, segments]
        distances **= 0.5
        from_vertices = segments + head_segment
        to_vertices = from_vertices + 1
        if out is None:
            return ProjectedPoints(
                projected_points[point_indices, segments, :],
                Locations(
                    from_vertices, to_vertices, fractions[point_indices, segments, 0]
                ),
                distances,
            )
        else:
            out.coords[...] = projected_points[point_indices, segments, :]
            out.locations.from_vertices[...] = from_vertices
            out.locations.to_vertices[...] = to_vertices
            out.locations.fractions[...] = fractions[point_indices, segments, 0]
            out.cartesian_distances[...] = distances
            return out

    def project_between_distances(
        self,
        d_from: float,
        d_to: float,
        distances: np.ndarray,
        extrapolate: bool = False,
        out: typing.Optional[ProjectedPoints] = None,
    ) -> ProjectedPoints:
        """Convenience method to project between two distances.

        `distances` are the vertex distances of the linestring

        Extrapolation only works if there are no repeated distances
        (simplify first to achieve this).
        """
        locations = locate(
            d_where=distances,
            d_what=np.array([d_from, d_to], dtype=float),
            extrapolate=extrapolate,
        )
        last_segment = self.fractions.shape[1] - 1
        return self.project(
            *location_to_single_segment(
                locations.from_vertices[0],
                locations.to_vertices[0],
                locations.fractions[0],
                prefer_zero_fraction=True,
                last_segment=last_segment,
            ),
            *location_to_single_segment(
                locations.from_vertices[1],
                locations.to_vertices[1],
                locations.fractions[1],
                prefer_zero_fraction=False,
                last_segment=last_segment,
            ),
            out,
        )


def interpolate(data: np.ndarray, locations: Locations) -> np.ndarray:
    """Interpolate data interpreted as a tensor-valued linestring.

    This is a generalization of shapely.geometry.LineString.interpolate.
    For fractions outside [0, 1], extrapolation will happen
    (clip first if you don't want this).

    Axis 0 in data has to be the vertex index axis.
    """
    # from + fraction * (to - from) without unecessary temporary arrays:
    n_extra_dims = len(data.shape) - 1
    if n_extra_dims:
        # numpy broadcasts from right to left but vertex index axis is leftmost
        # -1 means take all of the remaining space
        fractions = locations.fractions.reshape((-1,) + (1,) * n_extra_dims)
    else:
        fractions = locations.fractions
    data_from = data[locations.from_vertices]
    result = data[locations.to_vertices]  # advanced indexing already makes a copy
    result -= data_from
    result *= fractions
    result += data_from
    return result


def resample(
    x: np.ndarray, y: np.ndarray, x_prime: np.ndarray, extrapolate: bool = False
):
    """Convenience function for locating x_prime on x and interpolating y with those locations.

    This can be used as a replacement for scipy.interpolate.interp1d in the linear case
    when dealing with repeated x values (which scipy can't).

    NaNs in x_prime will pass through as NaNs in y_prime (the resampled result).
    """
    nan_mask = np.isnan(x_prime)
    nan_indices = np.nonzero(nan_mask)[0]
    if len(nan_indices):
        non_nan_indices = np.nonzero(np.logical_not(nan_mask))[0]
        y_prime_shape = (len(x_prime),) + y.shape[1:]
        if len(non_nan_indices):
            y_prime = np.empty(y_prime_shape)
            y_prime[nan_indices] = np.nan
            y_prime[non_nan_indices] = interpolate(
                y, locate(x, x_prime[non_nan_indices], extrapolate)
            )
            return y_prime
        else:
            return np.full(y_prime_shape, np.nan)
    else:
        return interpolate(y, locate(x, x_prime, extrapolate))


def substrings(
    data: np.ndarray,
    starts: Locations,
    ends: Locations,
) -> typing.List[np.ndarray]:
    """Extract substring copies of data interpeted as a tensor-valued linestring.

    This is a generalization of shapely.ops.substring.
    In contrast to shapely, two identical points are returned for a zero-length substring,
    the selection of the substring extent is different and many substrings are processed
    in one call.

    If an end lies before the corresponding start, a zero length substring from start to start
    is returned. Start/end points are interpolated/extrapolated linearly.
    If you don't want extrapolation, clip first.
    """
    # don't go in reverse direction
    ends = ends.max(starts)

    # the first indices that are not covered by start interpolation/extrapolation
    frac_start = starts.get_fractional_indices()
    frac_start_ceil = np.ceil(frac_start)
    first = frac_start_ceil.astype(int)
    first[frac_start == frac_start_ceil] += 1
    clip(first, starts.from_vertices, starts.to_vertices + 1, out=first)

    # the last indices that are not convered by end interpolation/extrapolation
    frac_end = ends.get_fractional_indices()
    frac_end_floor = np.floor(frac_end)
    last = frac_end_floor.astype(int)
    last[frac_end == frac_end_floor] -= 1
    clip(last, ends.from_vertices - 1, ends.to_vertices, out=last)

    data_length = len(data)
    start_data = interpolate(data, starts)
    end_data = interpolate(data, ends)
    substrings = []
    for fi, sdata, edata, la in zip(first, start_data, end_data, last):
        sli = slice(fi, la + 1)
        n_points = len(range(*sli.indices(data_length))) + 2
        sub_shape = (n_points,) + data.shape[1:]
        substring = np.empty(sub_shape)
        substring[0] = sdata
        substring[1:-1] = data[sli]
        substring[-1] = edata
        substrings.append(substring)

    return substrings


def locate(
    d_where: np.ndarray, d_what: np.ndarray, extrapolate: bool = False
) -> Locations:
    """Find locations of distances in a sequence of distances.

    `d_where` has to be sorted in ascending order (not checked).

    If an item of d_what exactly hits an item of d_where, the according location
    will be at fraction 0.5 of the virtual segment spanning all the occurences in d_where.
    E.g. if d_where is [1, 2, 3] and d_what is [2], location will be from 1 to 1 with fraction 0.5.
    And if d_where is [1, 2, 2, 3] and d_what is [2], location will be from 1 to 2
    with fraction 0.5.

    If `extrapolate` is set to `True` and an item `j` of `d_what` is smaller than any value of
    `d_where`, the location will go from 0 to i with `d_where[i] > `d_what[j]` with
    a negative fraction. If that is not possible, `ExtrapolationError` is raised.
    Similarly for extrapolating at the other end (then fraction is > 1).

    If `extrapolate` is set to `False` (default), the locations will be clipped at
    from/to fraction 0 and from/to last fraction 1.

    Attention: NaNs are sorted in at the end by numpy, so be careful to do your own filtering
    on NaNs in `d_what` (you will get a finite but probably meaningless result).
    """
    # d_where is sorted
    d_min = d_where[0]
    d_max = d_where[-1]

    from_vertices = np.searchsorted(d_where, d_what, side="left")
    to_vertices = np.searchsorted(d_where, d_what, side="right") - 1
    on_segment = from_vertices > to_vertices
    on_segment_indices = np.nonzero(on_segment)[0]
    # advanced indexing makes a copy, so this swap works
    from_vertices[on_segment_indices], to_vertices[on_segment_indices] = (
        to_vertices[on_segment_indices],
        from_vertices[on_segment_indices],
    )

    fractions = np.full_like(d_what, 0.5, dtype=float)
    if len(on_segment_indices):
        recalculate_on_segment_indices = False
        underflow_indices = np.nonzero(from_vertices == -1)[0]
        if len(underflow_indices):
            from_vertices[underflow_indices] = 0
            if extrapolate:
                to_vertex = np.searchsorted(d_where, d_min, side="right")
                if to_vertex == len(d_where):
                    raise ExtrapolationError(
                        "cannot extrapolate when all distances are the same"
                    )
                to_vertices[underflow_indices] = to_vertex
            else:
                fractions[underflow_indices] = 0.0
                on_segment[underflow_indices] = False
                recalculate_on_segment_indices = True
        overflow_indices = np.nonzero(to_vertices == len(d_where))[0]
        if len(overflow_indices):
            to_vertices[overflow_indices] = len(d_where) - 1
            if extrapolate:
                from_vertex = np.searchsorted(d_where, d_max, side="left") - 1
                if from_vertex == -1:
                    raise ExtrapolationError(
                        "cannot extrapolate when all distances are the same"
                    )
                from_vertices[overflow_indices] = from_vertex
            else:
                fractions[overflow_indices] = 1.0
                on_segment[overflow_indices] = False
                recalculate_on_segment_indices = True

        if recalculate_on_segment_indices:
            on_segment_indices = np.nonzero(on_segment)[0]

        if len(on_segment_indices):
            # advanced indexing makes a copy
            segment_fractions = d_what[on_segment_indices]
            if len(on_segment_indices) < len(d_what):
                filtered_from_vertices = from_vertices[on_segment_indices]
                filtered_to_vertices = to_vertices[on_segment_indices]
            else:
                # avoid copy
                filtered_from_vertices = from_vertices
                filtered_to_vertices = to_vertices
            d_where_from = d_where[filtered_from_vertices]
            segment_fractions -= d_where_from
            segment_fractions /= d_where[filtered_to_vertices] - d_where_from
            fractions[on_segment_indices] = segment_fractions
    return Locations(from_vertices, to_vertices, fractions)
