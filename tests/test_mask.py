"""Threshold masks, and region masks in pixel and metre coordinates."""

import numpy as np
import pytest

from pizprobe.mask import polygon, rectangle, threshold

# Non-square image and non-square pixels, so a swapped axis would fail.
SHAPE = (6, 8)
PIXEL = (0.5e-6, 2e-6)  # dx, dy in m


def box(rows, cols):
    out = np.zeros(SHAPE, dtype=bool)
    out[rows, cols] = True
    return out


Z = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, np.nan]])


def test_threshold_above_below_and_band():
    np.testing.assert_array_equal(threshold(Z, above=2.0), [[0, 0, 0], [1, 1, 0]])
    np.testing.assert_array_equal(threshold(Z, below=2.0), [[1, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(threshold(Z, above=0.5, below=3.5), [[0, 1, 1], [1, 0, 0]])


def test_threshold_bad_limits_raise():
    with pytest.raises(ValueError, match="at least one"):
        threshold(Z)
    with pytest.raises(ValueError, match="empty"):
        threshold(Z, above=3.0, below=1.0)


def test_rectangle_in_pixels():
    expected = box(slice(1, 4), slice(2, 5))
    np.testing.assert_array_equal(rectangle(SHAPE, x=(2, 4), y=(1, 3)), expected)
    np.testing.assert_array_equal(rectangle(SHAPE, x=(4, 2), y=(3, 1)), expected)


def test_one_pixel_in_metres():
    # Pixel z[row=1, col=3]: x = 3 * dx, y = (6 - 1 - 1) * dy, measured from bottom left.
    dx, dy = PIXEL
    x, y = 3 * dx, 4 * dy
    region = rectangle(SHAPE, x=(x, x), y=(y, y), pixel_size=PIXEL)
    np.testing.assert_array_equal(region, box(1, 3))


def test_rectangle_metres_matches_pixels():
    dx, dy = PIXEL
    # Rows 1..3 from the top are y = 4, 3, 2 pixels up from the bottom.
    metres = rectangle(SHAPE, x=(2 * dx, 4 * dx), y=(2 * dy, 4 * dy), pixel_size=PIXEL)
    np.testing.assert_array_equal(metres, rectangle(SHAPE, x=(2, 4), y=(1, 3)))


def test_polygon_square_matches_rectangle():
    # Vertices between pixel centres, so no centre is on an edge.
    square = [(1.5, 0.5), (4.5, 0.5), (4.5, 3.5), (1.5, 3.5)]
    np.testing.assert_array_equal(polygon(SHAPE, square), box(slice(1, 4), slice(2, 5)))


def test_polygon_metres_matches_pixels():
    dx, dy = PIXEL
    pixels = [(0.5, 0.5), (6.5, 1.5), (3.5, 4.5)]
    metres = [(c * dx, (SHAPE[0] - 1 - r) * dy) for c, r in pixels]
    np.testing.assert_array_equal(
        polygon(SHAPE, metres, pixel_size=PIXEL), polygon(SHAPE, pixels)
    )


def test_polygon_area():
    n = 400
    triangle = [(0, 0), (n - 1, 0), (0, n - 1)]
    area = polygon((n, n), triangle).sum()
    assert area == pytest.approx((n - 1) ** 2 / 2, rel=0.01)


def test_region_outside_image_is_empty():
    assert not rectangle(SHAPE, x=(20, 30), y=(0, 5)).any()


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="shape"):
        rectangle((5,), x=(0, 1), y=(0, 1))
    with pytest.raises(ValueError, match="pixel_size"):
        rectangle(SHAPE, x=(0, 1), y=(0, 1), pixel_size=(0, 1e-6))
    with pytest.raises(ValueError, match="at least 3"):
        polygon(SHAPE, [(0, 0), (1, 1)])
