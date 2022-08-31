import typing
import itertools
import logging
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike
from shapely.geometry import LineString
import pyproj

from . import EPSG4978


logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def get_trafo(
    from_crs: pyproj.CRS, to_crs: pyproj.CRS = EPSG4978, strict_axis_order: bool = False
) -> typing.Callable[
    [np.ndarray, np.ndarray, np.ndarray],
    typing.Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    return pyproj.Transformer.from_crs(
        from_crs,
        to_crs,
        always_xy=not strict_axis_order,
    ).transform


def transform_coords(
    coords: ArrayLike,
    trafo: typing.Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray],
    ],
    out: typing.Optional[np.ndarray] = None,
    skip_z_output: bool = False,
) -> np.ndarray:
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.shape[-1] != 3:
        raise ValueError("last axis has to be of length 3")
    if out is None:
        if skip_z_output:
            out = np.empty(coords_arr.shape[:-1] + (2,))
        else:
            out = np.empty_like(coords_arr)
    if skip_z_output:
        out[..., 0], out[..., 1], _ = trafo(
            coords_arr[..., 0], coords_arr[..., 1], coords_arr[..., 2]
        )
    else:
        out[..., 0], out[..., 1], out[..., 2] = trafo(
            coords_arr[..., 0], coords_arr[..., 1], coords_arr[..., 2]
        )
    return out


def simplify_2d_keep_rest(coords: ArrayLike, tolerance) -> np.ndarray:
    """Simplify in first two dimensions of linestring coords using Ramer-Douglas-Peucker algorithm.

    The other dimensions are not considered but they are kept intact for points which
    were not removed.
    """
    # TODO (nice to have): this is the only part in the project, where we depend on shapely.
    # Maybe we could implement our own variant of Ramer-Douglas-Peucker algorithm without
    # this ugly hack to get rid of shapely and to support 3d.

    coords_arr = array_chk(coords, ((2, None), (2, None)), dtype=float)

    # shapely ignores the third dimension but keeps it intact as long as it is finite everywhere
    # so we can sneak in indices there (as floats)
    shapely_input = np.empty((coords_arr.shape[0], 3))
    shapely_input[:, :2] = coords_arr[:, :2]
    shapely_input[:, 2] = np.arange(coords_arr.shape[0])

    shapely_output = np.array(
        LineString(shapely_input)
        .simplify(tolerance=tolerance, preserve_topology=False)
        .coords
    )
    if shapely_output.shape[1] != 3:
        raise RuntimeError("shapely simplify lost the z dimension")
    return coords_arr[shapely_output[:, 2].astype(int)]


def iter_consecutive_groups(
    integers: typing.Iterable[int],
) -> typing.Iterator[typing.List[int]]:
    """Iterate over groups of consecutive integers."""
    return (
        [item[1] for item in group]
        for _, group in itertools.groupby(
            enumerate(integers),
            lambda t: t[0] - t[1],
        )
    )


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
                        f"axis {i} has wrong length (expected at least {n_min}, got {n})"
                    )
            elif n_max is not None:
                if n > n_max:
                    raise ValueError(
                        f"axis {i} has wrong length (expected at most {n_max}, got {n})"
                    )
    if chk_finite and not np.all(np.isfinite(arr)):
        raise ValueError("input not finite")
    return arr
