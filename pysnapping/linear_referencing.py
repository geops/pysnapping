import itertools
import logging
from functools import partial

import pyproj
import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString, MultiPoint

from .ordering import order_ok

# pyproj < 2 has no way to create geod from crs
# EPSG4326_GEOD = pyproj.CRS.from_epsg(4326).get_geod()
EPSG4326_GEOD = pyproj.Geod(ellps="WGS84")

logger = logging.getLogger(__name__)


class DebugGeoms:
    def __init__(self, geoms, properties=None):
        self.geoms = geoms
        self.properties = properties

    def __str__(self):
        import json
        from shapely.geometry import mapping

        properties = self.properties
        if properties is None:
            properties = [{}] * len(self.geoms)

        collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": mapping(geom),
                }
                for geom, props in zip(self.geoms, properties)
            ],
        }

        return json.dumps(collection, indent=2)


def get_cartesian_line_string_dists(ls_coords: np.ndarray):
    dists = np.empty(len(ls_coords))
    dists[0] = 0
    dists[1:] = np.cumsum(((ls_coords[1:, :2] - ls_coords[:-1, :2]) ** 2).sum(1) ** 0.5)
    return dists


def get_geodesic_line_string_dists(ls_coords: np.ndarray, geod=EPSG4326_GEOD):
    dists = np.empty(len(ls_coords))
    dists[0] = 0
    # pyproj < 2 has no line_lengths method on Geod :(
    lons = ls_coords[:, 0]
    lats = ls_coords[:, 1]
    segment_lengths = geod.inv(
        lons1=lons[:-1], lats1=lats[:-1], lons2=lons[1:], lats2=lats[1:]
    )[2]
    dists[1:] = np.cumsum(segment_lengths)
    return dists


def split_ls_coords(ls_coords, ls_dists, split_dists):
    """Split LineString coords at dists using pre-computed LineString dists

    Usage hint: put ls_dists into ls_coords to automatically get dists for the parts back
    e.g. `split_ls_coords(ls_coords, ls_coords[:, 2], split_dists)`
    (`ls_coords` can be of higher dimensions than needed and the rest will be correctly
     interpolated)

    `ls_coords` has to be strictly increasing (not checked).
    `split_dists` has to be increasing.

    `split_dists` are clipped to `ls_coords[0]` / `ls_coords[-1]`


    Note: We cannot use `shapely.ops.substring` as a replacement or for the implementation
    for the following reasons:
    - it throws away any extra dimensions (but we need them later)
    - it only works with native dists (cartesian distances between LineString vertices)
      but we also have to deal with dists from external sources or geodesic dists in some cases
    - it is slow: pure python, no pre-computed dists, only a single split_dist
                  (we import several GiB of shapes)
    """

    if len(ls_coords) < 2:
        raise ValueError("bad length: too few coords")
    if len(ls_coords) != len(ls_dists):
        raise ValueError("bad length: ls_coords and ls_dists have different length")

    if np.any(np.diff(split_dists) < 0):
        raise ValueError("split_dists not increasing")

    if (
        len(split_dists) == 0
    ):  # split_dists might be an array, we cannot use `not split_dists`
        return [ls_coords]

    # np.asarray is the identity map if input is already an array with correct dtype
    ls_coords = np.asarray(ls_coords, dtype=float)
    ls_dists = np.asarray(ls_dists, dtype=float)
    split_dists = np.clip(split_dists, ls_dists[0], ls_dists[-1])

    # First we find for each split dist the last vertex index with
    # smaller vertex dist and the first vertex index with greater vertex dist
    # (using bisection search in the sorted ls_dists).
    # We do not allow for equality to avoid duplicate points when adding the interpolated
    # start/end points.

    # `side="left"` means sort item j in at i[j] s.th.
    # `ls_dists[i[j] - 1] < split_dists[j] <= ls_dists[i[j]]`
    last_smaller = np.searchsorted(ls_dists, split_dists, side="left") - 1
    # `side="right"` means sort item j in at i[j] s.th.
    # `ls_dists[i[j] - 1] <= split_dists[j] < ls_dists[i[j]]`
    first_greater = np.searchsorted(ls_dists, split_dists, side="right")

    # now calculate the interpolated coords between the parts
    # we have to include the vertices left/right of every point to be interpolated

    input_indices = np.empty(len(last_smaller) * 2, dtype=int)

    # clipping for special case: if split is exactly on first vertex,
    # we get last_smaller = -1 and we clip this to zero
    # (doesn't matter if we use the left or right side for interpolation in that case)
    np.clip(last_smaller, 0, None, out=input_indices[::2])

    # we do not use first_greater (fg) but last_smaller (ls) + 1 for the other side,
    # since there are two cases:
    # 1. split strictly between vertices:
    # ls      ls + 1 = fg
    # x-------x
    #     |
    #
    # 2. split stricly on vertex:
    # ls      ls + 1 != fg      fg
    # x-------x-----------------x
    #         |
    input_indices[1::2] = input_indices[::2] + 1

    # not sure if interp1d can deal with duplicate input points, so better filter them away
    input_indices = [g[0] for g in itertools.groupby(input_indices)]

    assert len(input_indices) >= 2, "implementation works as expected"
    interpolated_coords = interp1d(
        x=ls_dists[input_indices],
        y=ls_coords[input_indices],
        kind="linear",
        bounds_error=True,
        assume_sorted=True,
        axis=0,
    )(split_dists)

    # slices of vertices fully included in parts (without vertices on the boundaries)
    # (they can be empty if a part starts/ends at the same segment)
    part_slices = [slice(1, last_smaller[0] + 1)]
    part_slices.extend(
        slice(first_greater[i], last_smaller[i + 1] + 1)
        for i in range(len(split_dists) - 1)
    )
    part_slices.append(slice(first_greater[-1], len(ls_dists) - 1))

    extra_shape = ls_coords.shape[1:]  # we keep all irrelevant dimensions
    # use list, not np.array (unequal shape of nested sequences)
    # + 2 for interpolated start/end coords / vertices on the boundaries
    part_coords = [
        np.empty((max(s.stop - s.start, 0) + 2,) + extra_shape) for s in part_slices
    ]

    # set interpolated start/end coords
    part_coords[0][0] = ls_coords[0]
    for i, icoords in enumerate(interpolated_coords):
        part_coords[i][-1] = icoords
        part_coords[i + 1][0] = icoords
    part_coords[-1][-1] = ls_coords[-1]

    # set coords of vertices fully included
    for i, s in enumerate(part_slices):
        part_coords[i][1:-1] = ls_coords[s]

    return part_coords


class SnappingError(ValueError):
    pass


def get_metric_residua(line_string, multi_point, cartesian_mp_dists):
    multi_point_on_ls = MultiPoint(
        [line_string.interpolate(d) for d in cartesian_mp_dists]
    )
    # array interface for MultiPoint is deprecated in shapely >= 1.8 :(
    mp_coords = np.array([np.array(p.coords[0]) for p in multi_point.geoms])
    mp_on_ls_coords = np.array([np.array(p.coords[0]) for p in multi_point_on_ls.geoms])
    return EPSG4326_GEOD.inv(
        lons1=mp_coords[:, 0],
        lats1=mp_coords[:, 1],
        lons2=mp_on_ls_coords[:, 0],
        lats2=mp_on_ls_coords[:, 1],
    )[2]


def snap_points_in_order(
    line_string,
    ls_dists,
    multi_point,
    initial_mp_dists,
    min_dist,
    start_occupied,
    end_occupied,
    try_reverse,
    convergence_accuracy,
    get_residua=get_metric_residua,
    res_rtol=3,
    res_atol=200,
):
    """Snap points onto a line string, preserving linear referencing order

    If geoms are not in EPSG:4326, you have to supply a different `get_residua`.
    For the default of `get_residua`, `res_atol` is in meters.
    The residua are the distances between the snapped and original points.
    They are used in `np.allclose` to check whether residua after optimization
    are close to the optimal residua of the unordered solution. Default is
    to allow 3 times the optimal distance plus 200 meters.
    """

    if len(line_string.coords) != len(ls_dists) or len(multi_point.geoms) != len(
        initial_mp_dists
    ):
        raise ValueError("bad lengths")

    n_points = len(multi_point.geoms)

    if n_points == 0:
        return [], False

    start_dist = ls_dists[0] + start_occupied * min_dist
    end_dist = ls_dists[-1] - end_occupied * min_dist
    d_ok = partial(order_ok, v_min=start_dist, v_max=end_dist, d_min=min_dist)
    available_length = end_dist - start_dist
    needed_length = (n_points - 1) * min_dist
    if available_length < needed_length:
        raise SnappingError("line string too short")

    # these are the linear referencing dists shapely is working with
    cartesian_ls_dists = get_cartesian_line_string_dists(np.array(line_string.coords))

    from_cartesian = interp1d(
        x=cartesian_ls_dists,
        y=ls_dists,
        kind="linear",
        bounds_error=False,
        fill_value=(ls_dists[0], ls_dists[-1]),
        assume_sorted=True,
    )
    to_cartesian = interp1d(
        x=ls_dists,
        y=cartesian_ls_dists,
        kind="linear",
        bounds_error=False,
        fill_value=(cartesian_ls_dists[0], cartesian_ls_dists[-1]),
        assume_sorted=True,
    )

    # first try just snapping globally; if this works, we're done
    cartesian_mp_dists = [line_string.project(p) for p in multi_point.geoms]
    mp_dists = from_cartesian(cartesian_mp_dists)
    if d_ok(mp_dists):
        return mp_dists, False
    elif try_reverse and d_ok(mp_dists[::-1]):
        return mp_dists, True
    else:
        return _best_snap_points_in_order(
            line_string=line_string,
            multi_point=multi_point,
            cartesian_mp_dists=cartesian_mp_dists,
            initial_mp_dists=initial_mp_dists,
            min_dist=min_dist,
            start_dist=start_dist,
            end_dist=end_dist,
            ls_dists=ls_dists,
            convergence_accuracy=convergence_accuracy,
            from_cartesian=from_cartesian,
            to_cartesian=to_cartesian,
            try_reverse=try_reverse,
            res_rtol=res_rtol,
            res_atol=res_atol,
        )


def _best_snap_points_in_order(
    line_string,
    multi_point,
    cartesian_mp_dists,
    initial_mp_dists,
    min_dist,
    start_dist,
    end_dist,
    ls_dists,
    convergence_accuracy,
    from_cartesian,
    to_cartesian,
    try_reverse,
    res_rtol,
    res_atol,
):
    d_ok = partial(order_ok, v_min=start_dist, v_max=end_dist, d_min=min_dist)
    if not d_ok(initial_mp_dists):
        raise ValueError(
            f"initial mp dists are bad: start_dist={start_dist} "
            f"min_dist={min_dist} end_dist={end_dist} "
            f"first={initial_mp_dists[0]} min_diff={np.diff(initial_mp_dists).min()} "
            f"last={initial_mp_dists[-1]}"
        )
    optimal_residua = get_metric_residua(line_string, multi_point, cartesian_mp_dists)
    mp_dists, residuum = _snap_points_in_order(
        line_string=line_string,
        ls_dists=ls_dists,
        multi_point=multi_point,
        mp_dists=initial_mp_dists,
        convergence_accuracy=convergence_accuracy,
        from_cartesian=from_cartesian,
        to_cartesian=to_cartesian,
        start_dist=start_dist,
        end_dist=end_dist,
        min_dist=min_dist,
    )
    reverse = False
    if try_reverse:
        # retry with point order reversed
        alt, alt_residuum = _snap_points_in_order(
            line_string=line_string,
            ls_dists=ls_dists,
            multi_point=multi_point.geoms[::-1],  # reverse points
            mp_dists=initial_mp_dists,
            convergence_accuracy=convergence_accuracy,
            from_cartesian=from_cartesian,
            to_cartesian=to_cartesian,
            start_dist=start_dist,
            end_dist=end_dist,
            min_dist=min_dist,
        )
        if alt_residuum < residuum:
            # we reverse later after order_ok check
            reverse = True
            mp_dists = alt

    if not d_ok(mp_dists):
        raise AssertionError(
            f"bad mp_dists from _snap_points_in_order: start_dist={start_dist} "
            f"min_dist={min_dist} end_dist={end_dist} "
            f"first={mp_dists[0]} min_diff={np.diff(mp_dists).min()} last={mp_dists[-1]}"
        )

    if reverse:
        mp_dists = mp_dists[::-1]

    # guard against bad local optima (if this happens and the data is OK, we have to
    # improve the algorithm in _snap_points_in_order)
    residua = get_metric_residua(line_string, multi_point, to_cartesian(mp_dists))
    if not np.allclose(residua, optimal_residua, rtol=res_rtol, atol=res_atol):
        logger.debug("GeoJSON:")
        geoms = list(multi_point.geoms)
        props = [{"kind": "stop", "index": i} for i in range(len(multi_point.geoms))]
        geoms.extend(line_string.interpolate(d) for d in to_cartesian(mp_dists))
        props.extend(
            {
                "kind": "snapped stop",
                "index": i,
                "dist_traveled": mp_dists[i],
                "residuum": residua[i],
                "opt_residuum": optimal_residua[i],
            }
            for i in range(len(mp_dists))
        )
        geoms.append(line_string)
        props.append({"kind": "trip shape"})
        logger.debug(
            "\n%s\n",
            DebugGeoms(geoms, props),
        )
        raise SnappingError("snapped points are too far away from original points")

    return mp_dists, reverse


def _snap_points_in_order(
    line_string,
    ls_dists,
    multi_point,
    mp_dists,
    convergence_accuracy,
    from_cartesian,
    to_cartesian,
    start_dist,
    end_dist,
    min_dist,
    n_iter_max=1000,
):
    # start with an ordered initial guess and iteratively optimize the solution,
    # preserving the order in each step until converged
    #
    # x: mp_dists
    # |: split_dists
    # s: start
    # e: end
    # p: multi_point
    #
    #           p                      p
    #
    #         start_dist       <-min_dist->             end_dist
    #              v                                       v
    # s------------|-----x----|------------|----x----------|----------e
    #
    #                region 1              region 2
    #
    # In the above example, in the next step, both will move to the left,
    # then regions are recalculated and the right part can move to the left
    # a little further.
    #
    # TODO [TGSRVI-851] (nice to have or if severe errors occur):
    # This will not always give the globally best solution in terms of distance of
    # snapped points to original points. But it should suffice for now.
    # Reason: It is a greedy algorithm where points only move if they are not
    # in the optimum position of their next segment.
    # Some ideas for improvement:
    #
    # - only apply locally where order is messed up (we already treat the case where
    #   just snapping gives the correct order, but this can be generalized)
    # - better initial guess: snap and swap locally until order is correct
    # - run multiple times with random initial guess
    # - add interaction term between stops and run a global optimizer for the residuum like
    #   basin hopping or simulated annealing
    #   (Currently a snapped stop will not move any more if it is in the optimum
    #    of its segment, so if a locally optimal point blocks a locally non-optimal
    #    point, they will get stuck and not move as a whole even if a better solution
    #    exists where both have to move. What already works is that if the segment
    #    of a locally optimal point changes, revealing a better optimum elsewhere,
    #    the point will move again to this optimum.)

    last_mp_dists = np.empty(len(mp_dists), dtype=float)
    last_mp_dists.fill(np.inf)
    residuum = _get_residuum(line_string, multi_point, mp_dists)
    ls_coords = np.array(line_string.coords)
    split_dists = np.empty(2 * len(mp_dists))
    split_dists[0] = start_dist
    split_dists[-1] = end_dist
    n_iter = 0
    delta = float("inf")

    # close to convergence, we move half the last delta in the next step, so if we would
    # run forever, we would still move (delta * sum (1/2 ** n), n=1 to infinity) = delta,
    # thus in all following steps combined, we move at most delta
    # so `convergence_accuracy / 1.1` should be fine to detect convergence with desired accuracy
    while delta > convergence_accuracy / 1.1:
        n_iter += 1
        if n_iter > n_iter_max:
            raise SnappingError("not converged")
        mid_dists = 0.5 * (mp_dists[:-1] + mp_dists[1:])
        split_dists[1:-2:2] = mid_dists - 0.5 * min_dist
        split_dists[2:-1:2] = mid_dists + 0.5 * min_dist

        # avoid decreasing split_dists due to numerical noise
        # fixme: this is somehow ugly, better calculate intervals and clip those
        diffs = np.clip(np.diff(split_dists), 0, None)
        split_dists[1:] = np.cumsum(diffs)
        np.clip(split_dists, start_dist, end_dist, out=split_dists)

        logger.debug(
            "n_iter=%d: delta=%.2e residuum=%.2e mp_dists=%s split_dists=%s",
            n_iter,
            delta,
            residuum,
            mp_dists,
            split_dists,
        )
        regions = split_ls_coords(ls_coords, ls_dists, split_dists)[1::2]
        # TODO (nice to have): use our own projection which can deal with non-cartesian
        # dists, so we don't have to convert to LineString and we don't have to
        # convert distances all the time
        cartesian_mp_dists = [
            LineString(r).project(p) + to_cartesian(offset)
            for r, p, offset in zip(regions, multi_point.geoms, split_dists[::2])
        ]
        residuum = _get_residuum(line_string, multi_point, cartesian_mp_dists)
        new_mp_dists = from_cartesian(cartesian_mp_dists)
        delta = np.abs(new_mp_dists - mp_dists).max()
        mp_dists = new_mp_dists

    logger.debug("converged in %d iterations with residuum %.2e", n_iter, residuum)

    return mp_dists, residuum


def _get_residuum(line_string, multi_point, cartesian_mp_dists):
    return max(
        line_string.interpolate(d).distance(p)
        for d, p in zip(cartesian_mp_dists, multi_point.geoms)
    )
