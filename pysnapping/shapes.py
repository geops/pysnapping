import logging
import itertools
from collections import namedtuple
import sys
import typing

from shapely.geometry import LineString
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from .linear_referencing import (
    split_ls_coords,
    snap_points_in_order,
    get_geodesic_line_string_dists,
    SnappingError,
    EPSG4326_GEOD,
)
from .ordering import fix_sequence_with_missing_values, NoSolution

logger = logging.getLogger(__name__)


# do not change: use namedtuple (not user defined class) for memory and performance reasons

GeomWithDists = namedtuple(
    "GeomWithDists",
    ("geom", "dists", "original_dists_mask", "finite_dists_mask", "n_finite"),
)


def simplify_rdp_2d_keep_z(coords, tolerance, fake_nan=sys.float_info.max):
    """Simplify line string coords using Ramer-Douglas-Peucker algorithm.

    The z-dimension is not considered but it is kept intact for points which
    were not removed. Also deals correctly with NaNs in the z-dimension.
    +-inf is not allowed in the z-dimension.

    `fake_nan` is an arbitrary finite floating point number that should not
    occur in the z-dimension values.
    """
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


def make_shape(lon_lat_dists, tolerance=1e-5):
    """Convert lon_lat_dist rows to GeomWithDists

    Coordinates are simplified with `tolerance`.

    May return `None` if there are less than 2 input rows or the
    overall cartesian distance is shorter than or equal to `tolerance`.
    """
    if len(lon_lat_dists) < 2:
        logger.warning("skipping shape with less than 2 points")
        return None

    lon_lat_dists = simplify_rdp_2d_keep_z(lon_lat_dists, tolerance=tolerance)

    geom = LineString(lon_lat_dists[:, :2])
    if geom.length <= tolerance:
        logger.warning("ignoring too short shape")
        return None

    # make a copy (view would keep the whole array)
    dists = lon_lat_dists[:, 2].copy()
    original_dists_mask = np.logical_not(np.isnan(dists))
    finite_dists_mask = original_dists_mask.copy()
    return GeomWithDists(
        geom,
        dists,
        original_dists_mask,
        finite_dists_mask,
        finite_dists_mask.sum(),
    )


def shape_dists_trusted_at(shape: GeomWithDists, dists):
    """Return boolean array indicating where shape has trusted dists.

    E.g.: boolean array `a` will have `a[i]` `True` iff shape has trusted
    distance information at `dists[i]`. A shape is said to have trusted
    distance information at `d` if either `d` exactly hits a vertex and
    the vertex has trusted dist or `d` is stricly between two vertices
    and both vertices have trusted distances.
    """
    # see split_ls_coords for an explanation of np.searchsorted
    ls_dists = np.asarray(shape.dists, dtype=float)
    dists = np.asarray(dists, dtype=float)

    # terminate with False at boundary in case dists are outside ls_dists
    trusted = np.empty(len(shape.original_dists_mask) + 2, dtype=bool)
    trusted[0] = False
    trusted[1:-1] = shape.original_dists_mask
    trusted[-1] = False

    # index shifted by 1 due to boundaries
    last_smaller = np.searchsorted(ls_dists, dists, side="left")  # no shift: -1 + 1 = 0
    first_greater = np.searchsorted(ls_dists, dists, side="right") + 1

    lsp1 = last_smaller + 1
    ls_trusted = trusted[last_smaller]
    lsp1_trusted = trusted[lsp1]
    fg_trusted = trusted[first_greater]

    # we have to differentiate between on vertex and on segment
    on_vertex = lsp1 != first_greater
    on_segment = np.logical_not(on_vertex)
    return (on_vertex & lsp1_trusted) | (on_segment & ls_trusted & fg_trusted)


def convert_to_geodesic_dists(
    shape: GeomWithDists, points: GeomWithDists, geod=EPSG4326_GEOD
) -> typing.Tuple[GeomWithDists, GeomWithDists]:
    """Convert distances to geodesic distances along shape.

    Geometries are not copied.
    """
    finite_shape_dists = shape.dists[shape.finite_dists_mask]
    if np.any(np.diff(finite_shape_dists) <= 0):
        can_interpolate = False
        original_dists_mask = np.zeros_like(shape.original_dists_mask)
    else:
        can_interpolate = shape.n_finite >= 2
        original_dists_mask = shape.original_dists_mask.copy()
    geod_shape = GeomWithDists(
        shape.geom,
        get_geodesic_line_string_dists(np.array(shape.geom.coords)),
        original_dists_mask,
        np.ones_like(shape.finite_dists_mask),
        len(shape.finite_dists_mask),
    )

    if can_interpolate:
        dists = interp1d(
            shape.dists[shape.finite_dists_mask],
            geod_shape.dists[shape.finite_dists_mask],
            kind="linear",
            copy=False,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )(points.dists)
        finite_dists_mask = np.logical_not(np.isnan(dists))
        geod_points = GeomWithDists(
            points.geom,
            dists,
            points.original_dists_mask.copy(),
            finite_dists_mask,
            finite_dists_mask.sum(),
        )
    else:
        dists = np.empty_like(points.dists)
        dists.fill(np.nan)
        geod_points = GeomWithDists(
            points.geom,
            dists,
            np.zeros_like(points.finite_dists_mask),
            np.zeros_like(points.original_dists_mask),
            0,
        )

    return (geod_shape, geod_points)


def split_shape(
    shape: GeomWithDists,
    points: GeomWithDists,
    travel_times: typing.Optional[ArrayLike] = None,
    min_dist_meters=25,
    converge_meters=1,
    max_move_trusted_meters=15,
    try_reverse: bool = True,
    res_rtol: float = 3.0,
    res_atol: float = 200.0,
) -> typing.List[LineString]:
    """Split WGS84 shape at points

    If shape dists are not strictly increasing, they are discarded.

    Internally, all distances are converted to geodesic distances along the shape.
    Point distances are fixed by calling `estimate_missing_point_dists`.

    If dist is locally trusted for shape and a point,
    the split for this point occurs based only on dist.
    If points without trusted dist info lie between such splits,
    they are placed acoording to untrusted dists
    and then iteratively optimized such that they get closer to
    the given coords but the order is not destroyed.

    If try_reverse is set to True (default) and no point dist info is available at all,
    also the reverse order is considered (useful for GTFS data).

    Raises `linear_referencing.SnappingError` if snapping with desired
    parameters is not possible.

    min_dist_meters -- minimum distance between stops
    converge_meters -- stop iterative solution when geodesic dists along line string do not
                       move more than this
    max_move_trusted_meters -- tolerance for altered trusted point dists to still count as trusted
    res_rtol/res_atol -- Relative/absolute tolerance for residuum. Default means that a snapped stop
                         may be 3 times as far away from the original stop than the closest point on
                         the shape plus 200 meters.
                         If this is not the case, SnappingError is raised.
    """

    try_reverse = try_reverse and points.n_finite == 0
    n_points = len(points.dists)

    # converting to geodesic distances makes things much easier since
    # then we don't have to deal with missing shape distances any more
    # and we can directly work with the meter based parameters
    shape, points = convert_to_geodesic_dists(shape, points)
    assert shape.n_finite == len(shape.dists)

    points = estimate_missing_point_dists(shape, points, travel_times)
    assert points.n_finite == n_points

    # first step: split at trusted dists
    shape_trusted = shape_dists_trusted_at(shape, points.dists)
    trusted = shape_trusted & points.original_dists_mask
    split_indices = np.nonzero(trusted)[0]
    # 3rd dimension is for annotating with dists
    # (to get dists of parts after splitting)
    ls_coords = np.empty((len(shape.geom.coords), 3))
    ls_coords[:, :2] = shape.geom.coords
    ls_coords[:, 2] = shape.dists
    parts = split_ls_coords(ls_coords, shape.dists, points.dists[split_indices])

    # second step: Split at untrusted dists
    assert len(parts) == len(split_indices) + 1
    if len(split_indices) != n_points:
        parts = split_parts(
            parts=parts,
            split_indices=split_indices,
            points=points,
            min_dist_meters=min_dist_meters,
            try_reverse=try_reverse,
            converge_meters=converge_meters,
            res_rtol=res_rtol,
            res_atol=res_atol,
        )

    # there is no stop at the start/end of the first/last part -> slice away
    return [LineString(p[:, :2]) for p in itertools.islice(parts, 1, len(parts) - 1)]


def _get_estimator(
    x: np.ndarray, y: np.ndarray
) -> typing.Callable[[ArrayLike], ArrayLike]:
    # We have to eliminate duplicate consecutive values for the interpolation to work.
    x, indices = np.unique(x, return_index=True)
    if len(x) < 2:
        raise ValueError("too few unique values")
    y = y[indices]
    return interp1d(
        x,
        y,
        kind="linear",
        copy=False,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )


def _fix_travel_time_dists(mask, dists, d_min, d_max, min_dist_meters):
    n_points = len(dists)
    indices = np.nonzero(mask)[0]
    # every group of consecutive indices is an isolated problem
    for _, group in itertools.groupby(
        enumerate(indices),
        # this function is constant as long as indices increase with step 1, thus giving
        # us groups of consecutive indices
        lambda t: t[0] - t[1],
    ):
        group_indices = [item[1] for item in group]
        i_left = group_indices[0] - 1
        i_right = group_indices[-1] + 1
        v_min = dists[i_left] + min_dist_meters if i_left >= 0 else d_min
        v_max = dists[i_right] - min_dist_meters if i_right < n_points else d_max
        dists[group_indices] = fix_sequence_with_missing_values(
            values=dists[group_indices],
            v_min=v_min,
            v_max=v_max,
            d_min=min_dist_meters,
        )


def estimate_missing_point_dists(
    shape: GeomWithDists,
    points: GeomWithDists,
    travel_times: typing.Optional[ArrayLike],
    min_dist_meters: float = 25.0,
    max_move_trusted_meters: float = 15.0,
) -> GeomWithDists:
    """Estimate missing point distances.

    `shape` is expected to have all distance values present (finite) and stricly increasing.
    This can be achieved by applying `convert_to_geodesic_dists` on `shape` and `points` first
    with a simplified shape geometry.

    The order and minimum spacing between point dists is fixed automatically,
    minimizing the sum of squares of deviations to the input data.
    If dists change too much during this process, they are marked as untrusted.

    `travel_times` is an optional array-like containing numbers that give the departure/arrival time
    at the points/stops in an arbitrary but consistent unit with an arbitrary offset
    under the assumption that the vehicle would not wait at the stop (arrival = departure time).
    Or in other words, the pure cumulative travel times. It can contain NaNs/Nones.
    If `travel_times` is given and the first and last value are not missing and different
    from each other and the finite values are all in increasing order
    (not necessarily strictly increasing), `travel_times` is used to estimate
    missing point distances.
    Otherwise, missing point distances are filled equidistantly.
    All estimated point distances are marked as untrusted.

    `geom` attribute is not copied.
    """
    if shape.n_finite != len(shape.dists):
        raise ValueError("shape has missing distances")
    elif np.any(np.diff(shape.dists) <= 0):
        raise ValueError("shape distances not strictly increasing")

    n_points = len(points.dists)

    # If the first/last point have unknown dist, we make the assumption that they
    # lie at the start/end of the shape. This does not have to be true but it is the
    # most sensible estimate we can make. And we need at least two known dists, so
    # we have to make an assumption here.
    # The iterative solution will correct this later if it is wrong.
    dists = points.dists.copy()
    finite_dists_mask = points.finite_dists_mask.copy()
    original_dists_mask = points.original_dists_mask.copy()
    geom = points.geom  # no copy as written in docstring
    del points  # protect from using below
    if not finite_dists_mask[0]:
        finite_dists_mask[0] = True
        dists[0] = shape.dists[0]
        original_dists_mask[0] = False
    if not finite_dists_mask[-1]:
        finite_dists_mask[-1] = True
        dists[-1] = shape.dists[-1]
        original_dists_mask[0] = False

    try:
        fixed_dists = fix_sequence_with_missing_values(
            values=dists,
            v_min=shape.dists[0],
            v_max=shape.dists[-1],
            d_min=min_dist_meters,
        )
    except NoSolution as error:
        raise SnappingError(
            "shape too short / too many points / min dist too large"
        ) from error
    not_too_far_away = np.abs(fixed_dists - dists) <= max_move_trusted_meters
    original_dists_mask &= not_too_far_away
    dists = fixed_dists

    # Now the finite point dists are all in order and there is enough space between them
    # and enough space for the missing dists and we have at least two unique known dists.
    # This allows us to use the finite dists for interpolating the missing distances.

    missing_mask = np.logical_not(finite_dists_mask)

    # shortcut if no dists are missing
    if not np.any(missing_mask):
        return GeomWithDists(
            geom,
            dists,
            original_dists_mask,
            finite_dists_mask,
            n_points,
        )

    if travel_times is not None:
        travel_times_arr = np.asarray(travel_times, dtype=float)
        del travel_times
        if len(travel_times_arr) != n_points:
            # we should not ignore this since this is probably an error in the code
            # and not in the data
            raise ValueError("travel_times has wrong shape")
        # We tolerate zero travel times between stops since this often occurs due to
        # rounding to full minutes.
        # We later fix this by forcing min_dist_meters on the interpolated point distances.
        # If any travel time is however decreasing, we assume all travel times are garbage,
        # falling back to equidistant interpolation since there is no obvious way
        # to fix negative travel times.
        mask = np.logical_not(np.isnan(travel_times_arr))
        usable_travel_times = travel_times_arr[mask]
        if (
            mask[0]
            and mask[-1]
            and travel_times_arr[-1] - travel_times_arr[0] > 0
            and np.all(np.diff(usable_travel_times) >= 0)
        ):
            # We can only estimate where we have known travel time.
            # The rest of the missing dists will be filled later by the equidistant estimation
            output_mask = missing_mask & mask

            if np.any(output_mask):
                input_mask = finite_dists_mask & mask
                x = travel_times_arr[input_mask]
                y = dists[input_mask]
                x_prime = travel_times_arr[output_mask]
                y_prime = _get_estimator(x, y)(x_prime)
                dists[output_mask] = y_prime
                # dists estimated from travel times could be too close together
                # or too close to known dists
                _fix_travel_time_dists(
                    missing_mask,
                    dists,
                    shape.dists[0],
                    shape.dists[-1],
                    min_dist_meters,
                )
                missing_mask[output_mask] = False
                original_dists_mask[output_mask] = False
                finite_dists_mask[output_mask] = True

    # now interpolate the remaining missing dists equidistantly (i.e. using indices as x values)
    if np.any(missing_mask):
        x = np.nonzero(finite_dists_mask)[0]
        y = dists[finite_dists_mask]
        x_prime = np.nonzero(missing_mask)[0]
        y_prime = _get_estimator(x, y)(x_prime)
        dists[missing_mask] = y_prime
        original_dists_mask[missing_mask] = False
        finite_dists_mask[missing_mask] = True

    assert np.all(finite_dists_mask)
    return GeomWithDists(
        geom,
        dists,
        original_dists_mask,
        finite_dists_mask,
        n_points,
    )


def split_parts(
    parts,
    split_indices,
    points,
    min_dist_meters,
    try_reverse,
    converge_meters,
    res_rtol,
    res_atol,
):
    if len(parts) > 1 and try_reverse:
        raise ValueError("try_reverse only works for a single part")

    n_points = len(points.geom.geoms)

    if len(split_indices):
        slices = [slice(0, split_indices[0])]
        slices.extend(
            slice(split_indices[i] + 1, split_indices[i + 1])
            for i in range(len(split_indices) - 1)
        )
        slices.append(slice(split_indices[-1] + 1, n_points))
    else:
        slices = [slice(0, n_points)]
    new_parts = []
    for part, sli in zip(parts, slices):
        points_in_part = points.geom.geoms[sli]
        if len(points_in_part.geoms):
            ls_dists = part[:, 2]
            split_dists, reverse = snap_points_in_order(
                line_string=LineString(part[:, :2]),
                ls_dists=ls_dists,
                multi_point=points_in_part,
                initial_mp_dists=points.dists[sli],
                min_dist=min_dist_meters,
                start_occupied=part is not parts[0],
                end_occupied=part is not parts[-1],
                try_reverse=try_reverse,
                convergence_accuracy=converge_meters,
                res_rtol=res_rtol,
                res_atol=res_atol,
            )
            if reverse:
                sub_parts = split_ls_coords(part, ls_dists, split_dists[::-1])
                sub_parts = [p[::-1] for p in reversed(sub_parts)]
            else:
                sub_parts = split_ls_coords(part, ls_dists, split_dists)
            new_parts.extend(sub_parts)
        else:
            new_parts.append(part)
    return new_parts
