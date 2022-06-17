import itertools
import typing

import numpy as np
from numpy.typing import ArrayLike


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
