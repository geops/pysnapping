import typing
import itertools
import sys
import logging
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike
from shapely.geometry import LineString
import pyproj

from . import EPSG4326


logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def get_trafo(
    from_crs: pyproj.CRS, always_xy: bool
) -> typing.Callable[[np.ndarray, np.ndarray], typing.Tuple[np.ndarray, np.ndarray]]:
    return pyproj.Transformer.from_crs(
        from_crs,
        EPSG4326,
        always_xy=always_xy,
    ).transform


def transform_coords(
    coords: ArrayLike,
    trafo: typing.Callable[
        [np.ndarray, np.ndarray], typing.Tuple[np.ndarray, np.ndarray]
    ],
    out: typing.Optional[np.ndarray] = None,
) -> np.ndarray:
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.shape[-1] != 2:
        raise ValueError("last axis has to be of length 2")
    if out is None:
        out = np.empty_like(coords_arr)
    out[..., 0], out[..., 1] = trafo(coords_arr[..., 0], coords_arr[..., 1])
    return out


def simplify_2d_keep_z(
    coords: ArrayLike, tolerance, fake_nan=sys.float_info.max
) -> np.ndarray:
    """Simplify line string coords using Ramer-Douglas-Peucker algorithm.

    The z-dimension is not considered but it is kept intact for points which
    were not removed. Also deals correctly with NaNs in the z-dimension.
    +-inf is not allowed in the z-dimension.

    `fake_nan` is an arbitrary finite floating point number that should not
    occur in the z-dimension values.
    """
    # TODO (nice to have): this is the only part in the project, where we depend on shapely.
    # Maybe we could implement our own variant of Ramer-Douglas-Peucker algorithm without
    # this ugly hack to get rid of shapely.

    # shapely seems to silently loose the z-dimension in simplify
    # if there are NaNs or infs present :(
    coords_arr = array_chk(coords, ((2, None), 3), dtype=float)
    coords_arr[np.isnan(coords_arr[:, 2]), 2] = fake_nan

    if not np.all(np.isfinite(coords_arr[:, 2])):
        raise ValueError("+-inf not allowed in z dimension")

    simple_coords = np.array(
        LineString(coords_arr)
        .simplify(tolerance=tolerance, preserve_topology=False)
        .coords
    )
    if simple_coords.shape[1] != 3:
        raise RuntimeError("shapely simplify lost the z dimension")

    simple_coords[simple_coords[:, 2] == fake_nan, 2] = np.nan
    return simple_coords


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


def fix_repeated_x(x: ArrayLike, y: ArrayLike):
    """Remove repeated consecutive x values and average corresponding y values.

    `x` is assumed to be sorted in ascending order.
    Axis 0 of y is expected to correspond to x values.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    assert len(x_arr) == len(y_arr)
    n_points = len(x_arr)

    x_indices = [
        (x, list(indices))
        for x, indices in itertools.groupby(range(n_points), key=lambda i: x_arr[i])
    ]
    if all(len(indices) == 1 for _, indices in x_indices):
        # speedup if nothing is repeated
        return x_arr, y_arr
    else:
        new_n_points = len(x_indices)
        new_x = np.empty(new_n_points)
        new_y_shape: typing.Any = list(y_arr.shape)
        new_y_shape[0] = new_n_points
        new_y_shape = tuple(new_y_shape)
        new_y = np.empty(new_y_shape)
        new_x[...], new_y[...] = zip(
            *((x, y_arr[indices].mean(0)) for x, indices in x_indices)
        )
        return new_x, new_y


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
