"""analysis.profile and the line-cut measurements on synthetic images with a known surface."""

import numpy as np
import pytest

from pizprobe.analysis import Profile, crossings, markers, profile, step_height

# Non-square image and non-square pixels, so a swapped axis would fail.
ROWS, COLS = 40, 60
PIXEL = (0.5e-6, 2e-6)  # dx, dy in m
R, C = np.mgrid[0:ROWS, 0:COLS].astype(float)
PLANE = 1e-9 * C - 3e-9 * R  # height = 1 nm per column, -3 nm per row


def test_horizontal_and_vertical_lines_follow_the_plane():
    p = profile(PLANE, (5, 10), (45, 10))
    np.testing.assert_allclose(p.distance, np.arange(41))
    np.testing.assert_allclose(p.values, PLANE[10, 5:46], atol=1e-20)
    p = profile(PLANE, (7, 2), (7, 30))
    np.testing.assert_allclose(p.values, PLANE[2:31, 7], atol=1e-20)


def test_diagonal_line_is_straight_with_the_right_slope():
    # Bilinear interpolation is exact on a plane, even between pixel centres.
    p = profile(PLANE, (3.3, 4.1), (50.7, 33.9))
    dc, dr = 50.7 - 3.3, 33.9 - 4.1
    expected = PLANE[0, 0] + 1e-9 * (3.3 + p.distance / np.hypot(dc, dr) * dc) \
        - 3e-9 * (4.1 + p.distance / np.hypot(dc, dr) * dr)
    np.testing.assert_allclose(p.values, expected, atol=1e-20)
    assert np.all(np.diff(p.distance) <= 1 + 1e-12)
    assert p.distance[-1] == pytest.approx(np.hypot(dc, dr))


def test_metres_match_pixels_and_distance_uses_both_pixel_sizes():
    dx, dy = PIXEL
    # Pixel (col 5, row 30) to (col 45, row 10); in metres y counts up from the bottom row.
    start_m = (5 * dx, (ROWS - 1 - 30) * dy)
    end_m = (45 * dx, (ROWS - 1 - 10) * dy)
    in_px = profile(PLANE, (5, 30), (45, 10))
    in_m = profile(PLANE, start_m, end_m, pixel_size=PIXEL)
    np.testing.assert_allclose(in_m.values, in_px.values, atol=1e-20)
    assert in_m.distance[-1] == pytest.approx(np.hypot(40 * dx, 20 * dy))
    assert in_m.start_px == pytest.approx((5, 30)) and in_m.end_px == pytest.approx((45, 10))


def test_width_averages_noise_but_not_a_plane():
    np.testing.assert_allclose(
        profile(PLANE, (5, 20), (45, 20), width=5).values,
        profile(PLANE, (5, 20), (45, 20)).values,
        atol=1e-20,
    )
    noisy = np.random.default_rng(1).normal(0, 1e-9, (ROWS, COLS))
    one = profile(noisy, (5, 20), (45, 20)).values
    five = profile(noisy, (5, 20), (45, 20), width=5).values
    assert five.std() < 0.6 * one.std()  # about 1 / sqrt(5) = 0.45


def test_step_height_away_from_the_edge():
    step = np.where(C < 30, 100e-9, 0.0)
    p = profile(step, (5, 20), (55, 20), width=3)
    far = np.abs(p.distance - 25) > 2  # step between columns 29 and 30 (distance 24-25)
    assert set(np.round(p.values[far] * 1e9, 6)) == {0.0, 100.0}


def test_band_outside_the_image_is_left_out():
    p = profile(PLANE, (0, 0), (20, 0), width=3)   # band rows -1, 0, 1: row -1 is outside
    np.testing.assert_allclose(p.values, PLANE[0:2, 0:21].mean(axis=0), atol=1e-20)


def test_profile_is_a_small_record():
    p = profile(PLANE, (5, 10), (45, 10), pixel_size=None, width=3)
    assert isinstance(p, Profile)
    assert repr(p) == "<Profile 41 points, 40.0 px, width 3>"
    with pytest.raises(AttributeError):
        p.width = 5  # frozen


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="outside the image"):
        profile(PLANE, (5, 10), (70, 10))
    with pytest.raises(ValueError, match="same point"):
        profile(PLANE, (5, 10), (5, 10))
    with pytest.raises(ValueError, match="width"):
        profile(PLANE, (5, 10), (45, 10), width=0)
    with pytest.raises(ValueError, match="width"):
        profile(PLANE, (5, 10), (45, 10), width=2.5)
    with pytest.raises(ValueError, match="pixel_size"):
        profile(PLANE, (0, 0), (1e-6, 0), pixel_size=(0, 1e-6))


# --- markers ------------------------------------------------------------------------

def test_markers_interpolate_between_samples():
    p = profile(PLANE, (5, 10), (45, 10))           # 1 nm per px along the line
    m = markers(p, (2.5, 30))
    np.testing.assert_allclose(m.values, PLANE[10, 5] + 1e-9 * np.array([2.5, 30]))
    assert m.dx == pytest.approx(27.5) and m.dz == pytest.approx(27.5e-9)


def test_markers_across_a_step_in_metres():
    dx, dy = PIXEL
    step = np.where(C < 30, 100e-9, 0.0)
    p = profile(step, (5 * dx, 20 * dy), (55 * dx, 20 * dy), pixel_size=PIXEL)
    m = markers(p, (10 * dx, 40 * dx))              # top, then floor
    assert m.dz == pytest.approx(-100e-9) and m.dx == pytest.approx(30 * dx)
    # Both ends of the profile are allowed.
    markers(p, (0, p.distance[-1]))


def test_markers_bad_arguments_raise():
    p = profile(PLANE, (5, 10), (45, 10))
    with pytest.raises(ValueError, match="two numbers"):
        markers(p, (1, 2, 3))
    with pytest.raises(ValueError, match="outside the profile"):
        markers(p, (1, 41))
    with pytest.raises(ValueError, match="outside the profile"):
        markers(p, (-1, 5))
    holes = PLANE.copy()
    holes[10, 20] = np.nan
    with pytest.raises(ValueError, match="no finite value"):
        markers(profile(holes, (5, 10), (45, 10)), (15, 30))


# --- step_height --------------------------------------------------------------------

STEP = np.where(C < 30, 100e-9, 0.0)                # top at columns 0-29, floor from 30


def test_step_height_of_a_known_step():
    p = profile(STEP, (5, 20), (55, 20))            # distance d is column 5 + d
    s = step_height(p, a=(0, 20), b=(30, 50))       # top: columns 5-25, floor: 35-55
    np.testing.assert_allclose(s.levels, [100e-9, 0.0], atol=1e-20)
    assert s.dz == pytest.approx(-100e-9)
    np.testing.assert_array_equal(s.n, [21, 21])    # both ends included
    np.testing.assert_allclose(s.std, 0, atol=1e-20)


def test_step_height_ranges_in_either_order_and_in_metres():
    dx, dy = PIXEL
    p = profile(STEP, (5 * dx, 20 * dy), (55 * dx, 20 * dy), pixel_size=PIXEL)
    s = step_height(p, a=(50 * dx, 30 * dx), b=(20 * dx, 0))   # floor, then top
    assert s.dz == pytest.approx(100e-9)
    np.testing.assert_allclose(s.ranges, [[30 * dx, 50 * dx], [0, 20 * dx]])


def test_step_height_std_is_the_spread_and_nan_is_left_out():
    rough = STEP + np.where(C % 2 == 0, 1e-9, -1e-9)          # +-1 nm, mean 0
    rough[20, 10] = np.nan
    p = profile(rough, (5, 20), (55, 20))
    s = step_height(p, a=(0, 19), b=(30, 49))                  # 20 samples each
    np.testing.assert_allclose(s.levels, [100e-9, 0.0], atol=1e-10)
    # NaN samples near column 10 (distance 5) are left out of range a only.
    finite = np.isfinite(p.values)
    assert s.n[0] == finite[:20].sum() < 20 and s.n[1] == 20
    assert s.std[1] == pytest.approx(1e-9)


def test_step_height_bad_arguments_raise():
    p = profile(STEP, (5, 20), (55, 20))
    with pytest.raises(ValueError, match="a must be two numbers"):
        step_height(p, a=5, b=(30, 50))
    with pytest.raises(ValueError, match="b: distance 60 is outside the profile"):
        step_height(p, a=(0, 20), b=(30, 60))
    with pytest.raises(ValueError, match="Range a .* no finite profile sample"):
        step_height(p, a=(2.2, 2.8), b=(30, 50))               # between two samples


# --- crossings ----------------------------------------------------------------------

# Ridge with sloped sidewalls: 0 up to column 15, rising 20 nm per column to 100 nm at
# column 20, flat to column 40, falling back to 0 at column 45.
RIDGE = np.clip(np.minimum(C - 15, 45 - C) * 20e-9, 0, 100e-9)


def test_crossings_on_sloped_sidewalls_give_the_width_at_that_level():
    p = profile(RIDGE, (5, 20), (55, 20))           # distance d is column 5 + d
    half = crossings(p, 50e-9)
    np.testing.assert_allclose(half.distance, [12.5, 37.5])        # columns 17.5, 42.5
    foot = crossings(p, 20e-9)
    np.testing.assert_allclose(foot.distance, [11, 39])            # columns 16, 44
    # Wider near the foot, as for any sloped sidewall.
    assert np.diff(foot.distance)[0] > np.diff(half.distance)[0]


def test_crossings_at_a_sample_value_and_with_no_crossing():
    p = profile(RIDGE, (5, 20), (55, 20))
    top = crossings(p, 100e-9)                      # samples exactly at 100 nm count as above
    np.testing.assert_allclose(top.distance, [15, 35])             # columns 20 and 40
    none = crossings(p, 200e-9)
    assert none.distance.shape == (0,) and none.level == 200e-9


def test_crossings_skip_nan_and_work_in_metres():
    dx, dy = PIXEL
    holes = RIDGE.copy()
    holes[20, 42:44] = np.nan                       # hides the falling crossing at 50 nm
    p = profile(holes, (5 * dx, 19 * dy), (55 * dx, 19 * dy), pixel_size=PIXEL)
    np.testing.assert_allclose(crossings(p, 50e-9).distance, [12.5 * dx])


def test_crossings_bad_level_raises():
    p = profile(RIDGE, (5, 20), (55, 20))
    with pytest.raises(ValueError, match="finite number"):
        crossings(p, np.nan)
