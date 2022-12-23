import typing
import logging

import numpy as np
from numpy.typing import ArrayLike
import cvxpy

from . import NoSolution


logger = logging.getLogger(__name__)


def _check_params(
    values: ArrayLike, v_min: float, v_max: float, d_min: ArrayLike
) -> typing.Tuple[np.ndarray, np.ndarray]:
    values_arr = np.asarray(values, dtype=float)
    d_min_arr = np.asarray(d_min, dtype=float)
    if d_min_arr.ndim == 0:
        d_min_arr = np.empty((len(values_arr) - 1,))
        d_min_arr.fill(d_min)
    elif len(d_min_arr) != len(values_arr) - 1:
        raise ValueError("wrong array dimensions")
    if np.any(d_min_arr < 0):
        raise ValueError("d_min has to be >= 0 everywhere")
    if v_max < v_min:
        raise ValueError("v_max has to be >= v_min")
    return values_arr, d_min_arr


def order_ok(
    values: ArrayLike,
    v_min: float,
    v_max: float,
    d_min: ArrayLike,
    atol: typing.Optional[float] = None,
) -> bool:
    """Check if values fall inside [v_min, v_max] in the correct order and distance

    By default, 2% of the mean available space per value are tolerated as an absolute error.
    """
    values_arr, d_min_arr = _check_params(values, v_min, v_max, d_min)

    n_values = len(values_arr)
    if n_values == 0:
        return True

    if atol is None:
        atol = 0.02 * (v_max - v_min) / n_values
    return bool(
        values_arr[0] >= v_min - atol
        and values_arr[-1] <= v_max + atol
        and np.all(np.diff(values_arr) >= d_min_arr - atol)
    )


def fix_sequence(
    values: ArrayLike,
    v_min: float,
    v_max: float,
    d_min: ArrayLike,
    atol: typing.Optional[float] = None,
    **cvxpy_solve_args,
) -> np.ndarray:
    """Fix a sequence of numbers

    such that
      * all values are within [v_min, v_max]
      * each values[i + 1] is at least d_min[i] greater than its predecessor
      * the sum of squared distances to the original values is minimized

    Or in other words: fix order/minimum spacing, changing the values as little as possible.

    This can be formulated as a special quadratic programming problem which is
    solved using the cvxpy library.

    Raises NoSolution if there are too many values to fit between
    v_min and v_max with distances d_min.
    Raises cvxpy.SolverError if the optimal solution cannot be found.

    atol is the desired absolute accuracy which defaults to 2% of the mean available space per value
    """
    values_arr, d_min_arr = _check_params(values, v_min, v_max, d_min)
    n_values = len(values_arr)

    # shortcuts for trivial cases:
    if n_values == 0:
        return np.array([])
    elif n_values == 1:
        return np.clip(values_arr, v_min, v_max)
    available_length = v_max - v_min
    required_length = d_min_arr.sum()
    if atol is None:
        atol = 0.02 * available_length / n_values
    if required_length > available_length + 2 * atol:
        raise NoSolution(
            f"required length {required_length} > available length {available_length}"
        )
    elif required_length >= available_length - atol * n_values:
        # just enough space to fit all values
        # this could be a hard case for the solver, so we treat it seperately
        solution = np.empty_like(values_arr)
        solution.fill(v_min)
        scale = available_length / required_length
        solution[1:] += scale * np.cumsum(d_min_arr)
        return solution
    elif order_ok(values, v_min, v_max, d_min, atol):
        return values_arr

    # Still here? Non-trivial but existing solution:

    v = cvxpy.Variable(n_values)
    objective = cvxpy.Minimize(cvxpy.sum_squares(v - values_arr))
    constraints = [v[0] >= v_min, v[-1] <= v_max]
    constraints.extend(v[i + 1] >= v[i] + d_min_arr[i] for i in range(n_values - 1))
    problem = cvxpy.Problem(objective, constraints)

    if "eps_abs" not in cvxpy_solve_args and "eps_rel" not in "cvxpy_solve_args":
        cvxpy_solve_args["eps_abs"] = 0.5 * atol
        cvxpy_solve_args["eps_rel"] = 0

    # may raise cvxpy.SolverError
    problem.solve(**cvxpy_solve_args)

    if problem.status != cvxpy.OPTIMAL:
        raise cvxpy.SolverError(
            f"solver finished with non-optimal status {problem.status!r}"
        )

    solution = v.value

    logger.debug("quadratic programming solution: %s -> %s", values_arr, solution)

    return solution


def fix_sequence_with_missing_values(
    values: ArrayLike,
    v_min: float,
    v_max: float,
    d_min: float,
    atol: typing.Optional[float] = None,
    **cvxpy_solve_args,
) -> np.ndarray:
    """Like fix_sequence but with NaNs in values.

    The minimum distances are calculated from d_min and the
    number of NaNs between finite values, such that the NaNs also
    have enough space to be filled later.

    `None` is also accepted as `float("nan")`.
    """
    values_arr = np.asarray(values, dtype=float)
    available_indices = np.nonzero(np.logical_not(np.isnan(values_arr)))[0]

    if len(available_indices) == 0:
        available_length = v_max - v_min
        required_length = d_min * (len(values_arr) - 1)
        if required_length > available_length:
            raise NoSolution(
                f"required length {required_length} > available length {available_length}"
            )
        if values_arr is values:
            solution = values_arr.copy()
        else:
            solution = values_arr
    else:
        # space for missing values at the beginning
        v_min_tilde = v_min + d_min * available_indices[0]

        # space for missing values at the end
        v_max_tilde = v_max - d_min * (len(values_arr) - 1 - available_indices[-1])

        # space for missing values in between
        d_min_tilde = d_min * np.diff(available_indices)

        values_tilde = values_arr[available_indices]
        solution_tilde = fix_sequence(
            values_tilde,
            v_min_tilde,
            v_max_tilde,
            d_min_tilde,
            atol=atol,
            **cvxpy_solve_args,
        )
        solution = np.full_like(values_arr, np.nan)
        solution[available_indices] = solution_tilde

    return solution
