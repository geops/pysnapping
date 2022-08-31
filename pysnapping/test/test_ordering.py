import pytest
from numpy.testing import assert_allclose
import numpy as np

from pysnapping.ordering import fix_sequence, fix_sequence_with_missing_values, order_ok
from pysnapping import NoSolution


@pytest.mark.parametrize(
    "values,v_min,v_max,d_min,expected",
    [
        [[1, 3, 2, 4], 1, 4, 0, [1, 2.5, 2.5, 4]],
        [[1, 2, 2, 4], 1.1, 3.8, 0.5, [1.1, 1.75, 2.25, 3.8]],
        [[0, 1, 1, 3], 0, 3, 1, [0, 1, 2, 3]],
        [[0, 1, 1, 3], 0, 3, 0.9999, [0, 1, 2, 3]],
        [[0, 1, 1, 3], 0, 3, 0.999, [0, 1, 2, 3]],
        [[0, 1, 1, 3], 0, 3, 0.99, [0, 1, 2, 3]],
        [[0, 1, 1, 3], 0, 3, 0.9, [0, 0.9, 1.8, 3]],
        [list(range(101)), 10, 300, 2, list(range(10, 211, 2))],
    ],
)
def test_fix_sequence_same_d_min(values, v_min, v_max, d_min, expected):
    if d_min == 0:
        rtol = 1e-3
        atol = 1e-6
    else:
        rtol = 0
        atol = 0.02 * d_min
    assert not order_ok(values, v_min, v_max, d_min)
    fixed_values = fix_sequence(values, v_min, v_max, d_min)
    assert_allclose(fixed_values, expected, rtol=rtol, atol=atol)
    assert order_ok(fixed_values, v_min, v_max, d_min)


def test_fix_sequence_different_d_min_just_fit():
    values = [1, 3, 2, 7, 8]
    d_min = [1, 1, 1, 4]
    v_min = 1
    v_max = 8
    assert not order_ok(values, v_min, v_max, d_min)
    fixed_values = fix_sequence(values, v_min, v_max, d_min)
    expected = [1, 2, 3, 4, 8]
    assert_allclose(fixed_values, expected, rtol=0, atol=0.02)
    assert order_ok(fixed_values, v_min, v_max, d_min)


def test_fix_sequence_different_d_min_solver():
    values = [1, 3, 2, 7, 8]
    d_min = [1, 1, 1.2, 3.4]
    v_min = 1
    v_max = 8
    assert not order_ok(values, v_min, v_max, d_min)
    fixed_values = fix_sequence(values, v_min, v_max, d_min)
    expected = [1, 2, 3, 4.6, 8]
    assert_allclose(fixed_values, expected, rtol=0, atol=0.02)
    assert order_ok(fixed_values, v_min, v_max, d_min)


def test_fix_sequence_with_missing_values():
    values = [None, float("nan"), 1, 3, 2, None, 7, 7, None, None, 8.1, 31, np.nan]
    v_min = 0
    v_max = 30
    d_min = 1
    fixed_values = fix_sequence_with_missing_values(values, v_min, v_max, d_min)
    expected = np.array(
        [None, None, 2, 3, 4, None, 6, 7, None, None, 10, 29, None], dtype=float
    )
    assert_allclose(fixed_values, expected, rtol=0, atol=0.02)


def test_fix_sequence_with_only_missing_values():
    values = [None, None, None]
    v_min = 0
    v_max = 10

    with pytest.raises(NoSolution):
        fix_sequence_with_missing_values(values, v_min, v_max, d_min=5.001)

    for d_min in (4.9, 4.99, 4.999, 4.9999, 5):
        fixed_values = fix_sequence_with_missing_values(
            values, v_min, v_max, d_min=d_min
        )
        assert_allclose(fixed_values, [np.nan] * 3)
