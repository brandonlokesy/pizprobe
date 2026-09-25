"""Threshold masks, region masks in pixel and metre coordinates, and mask clean-up."""

import numpy as np
import pytest

from pizprobe.mask import grow, polygon, rectangle, remove_small, shrink, threshold

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


# --- grow / shrink / remove_small ---------------------------------------------------

def point(shape=(11, 11), at=(5, 5)):
    out = np.zeros(shape, dtype=bool)
    out[at] = True
    return out


def test_grow_is_a_disc():
    grown = grow(point(), 2)
    # Pixels with dx**2 + dy**2 <= 4 around the centre: 1 + 4 + 4 + 4 = 13.
    assert grown.sum() == 13
    assert grown[5, 3] and grown[5, 7] and grown[4, 4] and not grown[3, 3]


def test_shrink_undoes_grow_of_a_point():
    np.testing.assert_array_equal(shrink(grow(point(), 3), 3), point())


def test_shrink_removes_narrow_patches():
    stripe = np.zeros((11, 11), dtype=bool)
    stripe[:, 4:6] = True           # 2 px wide: gone after shrinking by 1 (needs 3 px)
    assert not shrink(stripe, 1).any()


def test_shrink_does_not_move_the_image_border():
    edge = np.zeros((11, 11), dtype=bool)
    edge[:, :4] = True              # touches the left border and the top and bottom
    out = shrink(edge, 1)
    np.testing.assert_array_equal(out[:, :3], True)
    assert not out[:, 3:].any()


def test_zero_pixels_returns_a_copy():
    m = point()
    for f in (grow, shrink):
        out = f(m, 0)
        np.testing.assert_array_equal(out, m)
        assert out is not m


def test_remove_small_keeps_large_patches():
    m = np.zeros((10, 10), dtype=bool)
    m[1, 1] = True                  # 1 px
    m[5:8, 5:8] = True              # 9 px
    np.testing.assert_array_equal(remove_small(m, 5), m & (np.arange(10)[:, None] > 3))


def test_remove_small_counts_corner_neighbours():
    m = np.zeros((5, 5), dtype=bool)
    m[1, 1] = m[2, 2] = m[3, 3] = True   # diagonal line: one patch of 3
    np.testing.assert_array_equal(remove_small(m, 3), m)


def test_fill_holes_with_inverted_mask():
    m = np.ones((7, 7), dtype=bool)
    m[3, 3] = False                 # 1 px hole
    np.testing.assert_array_equal(~remove_small(~m, 2), np.ones((7, 7), dtype=bool))


def test_morphology_bad_arguments_raise():
    with pytest.raises(ValueError, match="pixels"):
        grow(point(), -1)
    with pytest.raises(ValueError, match="pixels"):
        shrink(point(), 1.5)
    with pytest.raises(ValueError, match="min_pixels"):
        remove_small(point(), 0)
    with pytest.raises(ValueError, match="2-D"):
        grow(np.zeros(5, dtype=bool), 1)
