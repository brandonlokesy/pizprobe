"""plane_fit on synthetic images with a known surface."""

import numpy as np
import pytest

from pizprobe.processing import plane_fit

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
