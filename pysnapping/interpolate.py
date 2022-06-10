import itertools
import typing

import numpy as np
from numpy.typing import ArrayLike


def fix_repeated_x(x: ArrayLike, y: ArrayLike, axis: int):
    """Remove repeated consecutive x values and average corresponding y values.

    `x` is assumed to be sorted in ascending order.
    `axis` tells which axis of `y` corresponts to `x`.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_points = len(x_arr)

    xy = [
        (x, y_arr.take(list(indices), axis=axis).mean(axis))
        for x, indices in itertools.groupby(range(n_points), key=lambda i: x_arr[i])
    ]

    new_n_points = len(xy)
    new_x = np.empty(new_n_points)
    new_y_shape: typing.Any = list(y_arr.shape)
    new_y_shape[axis] = new_n_points
    new_y_shape = tuple(new_y_shape)
    new_y = np.empty(new_y_shape)
    # np.moveaxis gives a view of an array
    # we move the target axis to the beginning since our calculated y values are shaped like this
    new_x[...], np.moveaxis(new_y, axis, 0)[...] = zip(*xy)
    return new_x, new_y
