import logging
import itertools
from functools import partial
from collections import namedtuple
import sys

from shapely.geometry import LineString
import numpy as np
from scipy.interpolate import interp1d

from .linear_referencing import (
    split_ls_coords,
    snap_points_in_order,
    get_geodesic_line_string_dists,
)

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


def fix_dists(dists, get_alternative_measurements):
    """Fill missing dists (e.g. in GTFS shape_dist_traveled) if possible

    `alternative_measurements = get_alternative_measurements()`
    is called only if needed and at most once to get alternative measurements.

    The `alternative_measurements` help to fill gaps (this can be e.g. calculated distances
    for GTFS shapes.txt or travel times without waiting times for GTFS stop_times.txt).
    A locally affine linear dependency between `dists` and `alternative_measurements`
    is assumed for filling gaps.

    Return `(fixed_dists, original_dists_mask, n_finite)`.
    `original_dists_mask` is a boolean mask indicating where original
    and fixed dists are both finite.
    `n_finite` is the number of finite values in `dists`.
    It can be either `0`, `1` or `len(dists)`.
    """
    # always copy (we might write inplace)
    dists = np.array(dists, dtype=float)

    n_dists = len(dists)

    # np.isfinite gives a bool array of the same shape as the input
    dist_finite = np.isfinite(dists)
    n_dists_finite = dist_finite.sum()

    if np.any(np.diff(dists[dist_finite]) <= 0):
        logger.debug("dists not strictly increasing, removing all dists")
        dists.fill(np.nan)
        dist_finite.fill(False)
        after_fix_finite = dist_finite
        n_after_fix_finite = 0
    elif n_dists_finite != n_dists and n_dists_finite >= 2:
        alternative_measurements = np.array(get_alternative_measurements(), dtype=float)
        if len(alternative_measurements) != n_dists:
            raise ValueError("bad array length")
        am_finite = np.isfinite(alternative_measurements)
        if np.any(np.diff(alternative_measurements[am_finite]) <= 0):
            # in contrast to dists, we cannot tolerate this since this is
            # no external data but is calculated by us
            raise ValueError("alternative measurements not strictly increasing")

        # where we have known x (alt. measurement) and y (dist) data
        input_mask = dist_finite & am_finite
        if input_mask.sum() >= 2:
            logger.debug("filling gaps in dists")
            interpolating_fun = interp1d(
                x=alternative_measurements[input_mask],
                y=dists[input_mask],
                kind="linear",
                fill_value="extrapolate",  # extrapolate leading and trailing gaps
                assume_sorted=True,  # only for speed up (we have sorted x values)
            )
            dist_not_finite = np.logical_not(dist_finite)
            dists[dist_not_finite] = interpolating_fun(
                alternative_measurements[dist_not_finite]
            )
            after_fix_finite = np.ones_like(dists, dtype=bool)
            n_after_fix_finite = len(dists)
        else:
            logger.debug("cannot fill gaps, too few known points")
            after_fix_finite = dist_finite
            n_after_fix_finite = n_dists_finite
    else:
        after_fix_finite = dist_finite
        n_after_fix_finite = n_dists_finite

    return (dists, dist_finite, after_fix_finite, n_after_fix_finite)


def make_shape(lon_lat_dists, tolerance=1e-5):
    """Convert lon_lat_dist rows to GeomWithDists

    Coordinates are simplified with `tolerance` and missing distances are
    fixed (if possible).

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

    dists, original_dists_mask, finite_dists_mask, n_finite = fix_dists(
        lon_lat_dists[:, 2],
        get_alternative_measurements=partial(
            get_geodesic_line_string_dists, lon_lat_dists[:, :2]
        ),
    )

    return GeomWithDists(geom, dists, original_dists_mask, finite_dists_mask, n_finite)


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


def split_shape(
    shape: GeomWithDists,
    points: GeomWithDists,
    min_dist_meters=25,
    converge_meters=1,
):
    """Split shape at points

    If original dist is locally available for shape and point,
    the split for the point occurs based only on dist.
    If points without reliable dist info lie between such splits,
    they are placed equidistantly or acoording to untrusted dists
    and then iteratively optimized such that they get closer to
    the given coords but the order is not destroyed.
    The untrusted dists should not influence the result but they can
    reduce the number of steps needed for the iterative solution.

    If no dist info is available at all, also the reverse order is
    considered (GTFS allows shapes to be undirected, if shape_dist_traveled is not given).

    min_dist_meters -- minimum distance between stops (not checked for original shape_dist_traveled)
    converge_meters -- stop iterative solution when geodesic dists along line string do not
                       move more than this
    """

    # first step: split at trusted dists
    # (where shape_dist_traveled is the original value for both shapes.txt and stop_times.txt)
    n_points = len(points.geom.geoms)
    parts = []
    split_indices = []
    untrusted_dists = np.empty(n_points)
    untrusted_dists.fill(np.nan)
    try_reverse = False
    if shape.n_finite == 0 or points.n_finite == 0:
        logger.debug("n_finite=0")
        parts.append(np.array(shape.geom.coords))
        # if no shape_dist_traveled is given, shape can be in the wrong orientation
        try_reverse = True
    elif shape.n_finite == 1:
        logger.debug("n_finite=1")
        ls_coords = np.array(shape.geom.coords)
        finite_indices = np.nonzero(shape.finite_dists_mask)[0]
        assert len(finite_indices) == 1
        finite_index = finite_indices[0]
        finite_dist = shape.dists[finite_index]
        matching_indices = np.nonzero(points.dists == finite_dist)[0]
        if matching_indices:
            assert len(matching_indices) == 1
            matching_index = matching_indices[0]
            if (
                points.original_dists_mask[matching_index]
                and shape.original_dists_mask[finite_index]
            ):
                parts = [ls_coords[: finite_index + 1], ls_coords[finite_index:]]
                split_indices = [matching_index]
    elif shape.n_finite == len(shape.dists):
        logger.debug("n_finite=len(dists)")
        masked_indices = np.nonzero(points.finite_dists_mask)[0]
        masked_dists = points.dists[masked_indices]
        masked_shape_trusted = shape_dists_trusted_at(shape, masked_dists)
        masked_point_trusted = points.original_dists_mask[masked_indices]
        masked_trusted = masked_shape_trusted & masked_point_trusted
        trusted_dists = masked_dists[masked_trusted]
        untrusted_dist_indices = masked_indices[np.logical_not(masked_trusted)]
        untrusted_dists[untrusted_dist_indices] = points.dists[untrusted_dist_indices]
        if len(untrusted_dist_indices):
            # 3rd dimension is for annotating with dists
            # (to get dists of parts after splitting)
            ls_coords = np.empty((len(shape.geom.coords), 3))
            ls_coords[:, :2] = shape.geom.coords
            ls_coords[:, 2] = shape.dists
        else:
            ls_coords = np.array(shape.geom.coords)
        parts = split_ls_coords(ls_coords, shape.dists, trusted_dists)
        split_indices = masked_indices[masked_trusted]
    else:
        raise ValueError(
            f"unexpected number of finte dists: {shape.n_finite} / {len(shape.dists)}"
        )

    # Second step: Split at untrusted and unknown dists
    # (where shape_dist_traveled is interpolated/extrapolated
    # for shapes.txt and/or stop_times.txt or unknown)
    assert len(parts) == len(split_indices) + 1
    if len(split_indices) != n_points:
        parts = split_parts(
            parts=parts,
            split_indices=split_indices,
            points=points,
            untrusted_dists=untrusted_dists,
            min_dist_meters=min_dist_meters,
            try_reverse=try_reverse,
            converge_meters=converge_meters,
        )

    # there is no stop at the start/end of the first/last part -> slice away
    return [LineString(p[:, :2]) for p in itertools.islice(parts, 1, len(parts) - 1)]


def split_parts(
    parts,
    split_indices,
    points,
    untrusted_dists,
    min_dist_meters,
    try_reverse,
    converge_meters,
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
            # we use geodesic distances to be able to express min dist in meters
            geod_ls_dists = get_geodesic_line_string_dists(part)
            # if we have any point dist hints, we have to convert them to geodesic
            gtfs_point_dists_in_part = untrusted_dists[sli]
            geod_point_dists_in_part = np.array(gtfs_point_dists_in_part)
            indices = np.nonzero(np.logical_not(np.isnan(gtfs_point_dists_in_part)))[0]
            if len(indices):
                trafo = interp1d(
                    x=part[:, 2],  # we stored gtfs dists there in this case
                    y=geod_ls_dists,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(geod_ls_dists[0], geod_ls_dists[-1]),
                    assume_sorted=True,
                )
                geod_point_dists_in_part[indices] = trafo(
                    gtfs_point_dists_in_part[indices]
                )
            geod_split_dists, reverse = snap_points_in_order(
                line_string=LineString(part[:, :2]),
                ls_dists=geod_ls_dists,
                multi_point=points_in_part,
                initial_mp_dists=geod_point_dists_in_part,
                min_dist=min_dist_meters,
                start_occupied=part is not parts[0],
                end_occupied=part is not parts[-1],
                try_reverse=try_reverse,
                convergence_accuracy=converge_meters,
            )
            if reverse:
                sub_parts = split_ls_coords(part, geod_ls_dists, geod_split_dists[::-1])
                sub_parts = [p[::-1] for p in reversed(sub_parts)]
            else:
                sub_parts = split_ls_coords(part, geod_ls_dists, geod_split_dists)
            new_parts.extend(sub_parts)
        else:
            new_parts.append(part)
    return new_parts
