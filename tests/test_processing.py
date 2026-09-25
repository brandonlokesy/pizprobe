"""plane_fit and line_flatten on synthetic images with a known surface."""

import numpy as np
import pytest

from pizprobe.processing import line_flatten, plane_fit

# Non-square, so a swapped row/column axis would fail.
ROWS, COLS = 60, 80
Y, X = np.mgrid[0:ROWS, 0:COLS].astype(float)
PLANE = 2e-9 + 3e-11 * X - 5e-11 * Y
BOWL = 1e-12 * ((X - 30) ** 2 + (Y - 20) ** 2)
BOX = np.zeros((ROWS, COLS), dtype=bool)
BOX[20:35, 30:55] = True
H = 50e-9


def test_plane_is_removed():
    np.testing.assert_allclose(plane_fit(PLANE), 0, atol=1e-20)


def test_bowl_needs_order_two():
    z = PLANE + BOWL
    np.testing.assert_allclose(plane_fit(z, order=2), 0, atol=1e-20)
    assert np.ptp(plane_fit(z, order=1)) > 1e-9


def test_order_zero_subtracts_mean():
    np.testing.assert_allclose(plane_fit(PLANE, order=0), PLANE - PLANE.mean(), atol=1e-20)


def test_mask_keeps_feature_out_of_fit():
    z = PLANE + H * BOX
    out = plane_fit(z, mask=BOX)
    np.testing.assert_allclose(out[~BOX], 0, atol=1e-20)
    np.testing.assert_allclose(out[BOX], H, rtol=1e-9)
    # Without the mask the feature pulls the fit and the floor is no longer flat.
    assert np.ptp(plane_fit(z)[~BOX]) > 1e-9


def test_non_finite_points_are_skipped():
    z = PLANE.copy()
    z[5, 5] = np.nan
    out = plane_fit(z)
    assert np.isnan(out[5, 5])
    np.testing.assert_allclose(np.delete(out.ravel(), 5 * COLS + 5), 0, atol=1e-20)


def test_input_is_not_modified():
    z = PLANE.copy()
    plane_fit(z, mask=BOX)
    np.testing.assert_array_equal(z, PLANE)


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="2-D"):
        plane_fit(np.zeros(5))
    with pytest.raises(ValueError, match="mask shape"):
        plane_fit(PLANE, mask=np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="order"):
        plane_fit(PLANE, order=-1)
    with pytest.raises(ValueError, match="order"):
        plane_fit(PLANE, order=1.5)


def test_too_few_points_raise():
    with pytest.raises(ValueError, match="unmasked points"):
        plane_fit(PLANE, mask=np.ones((ROWS, COLS), dtype=bool))


def test_points_on_one_column_raise():
    mask = np.ones((ROWS, COLS), dtype=bool)
    mask[:, 10] = False  # floor is a single column: slope along x is undefined
    with pytest.raises(ValueError, match="one row or column"):
        plane_fit(PLANE, mask=mask)


# --- line_flatten -------------------------------------------------------------------

RNG = np.random.default_rng(0)
LINE_OFFSETS = RNG.normal(0, 2e-9, ROWS)[:, None]
LINE_SLOPES = RNG.normal(0, 5e-11, ROWS)[:, None]


def test_order_zero_removes_line_offsets():
    # A plane minus its per-line mean: the tilt along the lines stays.
    expected = PLANE - PLANE.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(line_flatten(PLANE + LINE_OFFSETS), expected, atol=1e-20)
    np.testing.assert_allclose(line_flatten(LINE_OFFSETS + 0 * X), 0, atol=1e-20)


def test_order_one_removes_offsets_and_slopes():
    z = LINE_OFFSETS + LINE_SLOPES * X
    np.testing.assert_allclose(line_flatten(z, order=1), 0, atol=1e-20)
    # Order 0 keeps the slope along each line.
    assert np.ptp(line_flatten(z, order=0)) > 1e-10


def test_mask_keeps_feature_out_of_line_fit():
    z = LINE_OFFSETS + LINE_SLOPES * X + H * BOX
    out = line_flatten(z, mask=BOX, order=1)
    np.testing.assert_allclose(out[~BOX], 0, atol=1e-20)
    np.testing.assert_allclose(out[BOX], H, rtol=1e-9)
    # Without the mask the lines through the feature are pulled down next to it.
    assert np.ptp(line_flatten(z, order=1)[~BOX]) > 1e-9


def test_order_one_absorbs_any_plane():
    z = PLANE + LINE_OFFSETS + H * BOX
    np.testing.assert_allclose(
        line_flatten(plane_fit(z, mask=BOX), mask=BOX, order=1),
        line_flatten(z, mask=BOX, order=1),
        atol=1e-20,
    )


def test_line_without_enough_points_is_left_and_warned():
    z = PLANE + LINE_OFFSETS
    mask = np.zeros((ROWS, COLS), dtype=bool)
    mask[3] = True                  # no points: skipped at order 0 and 1
    mask[7, 1:] = True              # one point: enough for order 0, not for order 1
    with pytest.warns(UserWarning, match="1 of 60 scan lines"):
        out0 = line_flatten(z, mask=mask, order=0)
    np.testing.assert_array_equal(out0[3], z[3])
    np.testing.assert_allclose(out0[7], z[7] - z[7, 0], atol=1e-20)
    with pytest.warns(UserWarning, match="2 of 60 scan lines"):
        out1 = line_flatten(z, mask=mask, order=1)
    np.testing.assert_array_equal(out1[[3, 7]], z[[3, 7]])


def test_line_flatten_skips_non_finite_points():
    z = LINE_OFFSETS + LINE_SLOPES * X
    z[5, 5] = np.nan
    out = line_flatten(z, order=1)
    assert np.isnan(out[5, 5])
    np.testing.assert_allclose(np.delete(out.ravel(), 5 * COLS + 5), 0, atol=1e-20)


def test_line_flatten_bad_arguments_raise():
    with pytest.raises(ValueError, match="2-D"):
        line_flatten(np.zeros(5))
    with pytest.raises(ValueError, match="order"):
        line_flatten(PLANE, order=2)
    with pytest.raises(ValueError, match="mask shape"):
        line_flatten(PLANE, mask=np.zeros((2, 2), dtype=bool))


def test_line_flatten_input_is_not_modified():
    z = PLANE + LINE_OFFSETS
    before = z.copy()
    line_flatten(z, mask=BOX, order=1)
    np.testing.assert_array_equal(z, before)


def test_median_ignores_unmasked_bumps():
    z = LINE_OFFSETS + 0 * X
    z[:, :5] += 30e-9               # a bump on every line, not masked (5 of 80 points)
    by_median = line_flatten(z, statistic="median")
    np.testing.assert_allclose(by_median[:, 5:], 0, atol=1e-20)
    # The mean is pulled up by the bump, so the floor ends up below zero.
    by_mean = line_flatten(z, statistic="mean")
    np.testing.assert_allclose(by_mean[:, 5:], -30e-9 * 5 / 80, atol=1e-20)


def test_median_skips_lines_like_the_mean():
    mask = np.zeros((ROWS, COLS), dtype=bool)
    mask[3] = True
    z = PLANE + LINE_OFFSETS
    with pytest.warns(UserWarning, match="1 of 60 scan lines"):
        out = line_flatten(z, mask=mask, statistic="median")
    np.testing.assert_array_equal(out[3], z[3])


def test_statistic_bad_arguments_raise():
    with pytest.raises(ValueError, match="statistic"):
        line_flatten(PLANE, statistic="mode")
    with pytest.raises(ValueError, match="only defined for order=0"):
        line_flatten(PLANE, order=1, statistic="median")
