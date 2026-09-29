"""plotting.image, plotting.histogram and plotting.profiles, drawn off-screen."""

import matplotlib

matplotlib.use("Agg")  # no window; must come before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pizprobe import plotting
from pizprobe.analysis import crossings, markers, profile, step_height
from pizprobe.mask import rectangle

SHAPE = (6, 8)
PIXEL = (0.5e-6, 2e-6)
Z = np.arange(48, dtype=float).reshape(SHAPE) * 1e-9
MASK = np.zeros(SHAPE, dtype=bool)
MASK[1:3, 2:5] = True


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.mark.parametrize("draw", [plotting.image, plotting.histogram])
def test_returns_fig_and_ax(ax, draw):
    # Given axes: drawn there, and its own figure is returned.
    fig, returned = draw(Z, ax=ax)
    assert returned is ax and fig is ax.get_figure()
    # No axes: a new figure, not the current one.
    before = plt.get_fignums()
    fig, new_ax = draw(Z)
    assert new_ax is not ax and fig.number not in before
    plt.close(fig)


def test_bad_input_leaves_no_empty_figure():
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="pixel_size"):
        plotting.image(Z, pixel_size=(0, 1e-6))
    assert plt.get_fignums() == before


def test_image_in_pixels(ax):
    plotting.image(Z, ax=ax)
    (shown,) = ax.images
    # Default imshow extent: pixel centres on integer columns / rows, row 0 at the top.
    assert shown.get_extent() == pytest.approx([-0.5, 7.5, 5.5, -0.5])


def test_image_extent_puts_pixel_centres_at_mask_coordinates(ax):
    dx, dy = PIXEL
    plotting.image(Z, pixel_size=PIXEL, ax=ax)
    left, right, bottom, top = ax.images[0].get_extent()
    # Centre of column c: left + (c + 0.5) * dx; of row r: top - (r + 0.5) * dy.
    c, r = 3, 1
    x, y = left + (c + 0.5) * dx, top - (r + 0.5) * dy
    assert (x, y) == pytest.approx((c * dx, (SHAPE[0] - 1 - r) * dy))
    # The same point selects that pixel in mask.rectangle.
    assert rectangle(SHAPE, x=(x, x), y=(y, y), pixel_size=PIXEL)[r, c]


def test_image_draws_mask_on_top(ax):
    plotting.image(Z, mask=MASK, ax=ax)
    base, overlay = ax.images
    shown = overlay.get_array()
    np.testing.assert_array_equal(~np.ma.getmaskarray(shown), MASK)


def test_histogram_lines_and_mask(ax):
    plotting.histogram(Z, mask=MASK, thresholds=[10e-9, 20e-9], ax=ax)
    assert len(ax.lines) == 2
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["all points", "masked"]


def test_bad_mask_raises(ax):
    with pytest.raises(ValueError, match="mask shape"):
        plotting.image(Z, mask=np.zeros((2, 2), dtype=bool), ax=ax)
    with pytest.raises(ValueError, match="mask shape"):
        plotting.histogram(Z, mask=np.zeros((2, 2), dtype=bool), ax=ax)


# --- profiles -----------------------------------------------------------------------

BIG = np.add.outer(np.arange(20.0), np.arange(30.0)) * 1e-9


def cuts(pixel_size=None):
    ends = [((2, 3), (25, 3)), ((4, 15), (20, 5))]
    if pixel_size is not None:
        dx, dy = pixel_size
        ends = [((c0 * dx, (19 - r0) * dy), (c1 * dx, (19 - r1) * dy))
                for (c0, r0), (c1, r1) in ends]
    return [profile(BIG, s, e, pixel_size=pixel_size) for s, e in ends]


@pytest.mark.parametrize("image_px", [None, PIXEL])
@pytest.mark.parametrize("profile_px", [None, PIXEL])
def test_profiles_draws_matching_lines_and_curves(image_px, profile_px):
    fig, (ax_image, ax_profile) = plotting.profiles(BIG, cuts(profile_px), pixel_size=image_px)
    assert len(ax_image.lines) == 2 and len(ax_profile.lines) == 2
    for drawn, curve in zip(ax_image.lines, ax_profile.lines):
        assert drawn.get_color() == curve.get_color()
    # Line 2 starts at pixel (col 4, row 15), whatever units the profiles and axes use.
    x, y = ax_image.lines[1].get_xydata()[0]
    expected = (4, 15) if image_px is None else (4 * PIXEL[0], (19 - 15) * PIXEL[1])
    assert (x, y) == pytest.approx(expected)
    plt.close(fig)


@pytest.mark.parametrize("image_px", [None, PIXEL])
def test_profiles_draws_band_for_width(image_px):
    wide = profile(BIG, (2, 10), (25, 10), width=5)
    fig, (ax_image, _) = plotting.profiles(BIG, [wide, cuts()[0]], pixel_size=image_px)
    # One strip, for the wide profile only, in the colour of its line.
    (band,) = ax_image.patches
    assert band.get_facecolor()[:3] == matplotlib.colors.to_rgb(ax_image.lines[0].get_color())
    # Horizontal line at row 10, width 5: rows 7.5 to 12.5, columns 2 to 25.
    corners = band.get_xy()[:4]
    expected = np.array([(2, 7.5), (25, 7.5), (25, 12.5), (2, 12.5)])
    if image_px is not None:
        expected = np.column_stack([expected[:, 0] * PIXEL[0], (19 - expected[:, 1]) * PIXEL[1]])
    np.testing.assert_allclose(sorted(map(tuple, corners)), sorted(map(tuple, expected)))
    plt.close(fig)


def test_profiles_band_does_not_widen_image_axes():
    edge = profile(BIG, (0, 0), (29, 0), width=7)       # half the strip is off the image
    fig, (ax_image, _) = plotting.profiles(BIG, [edge])
    assert ax_image.get_xlim() == pytest.approx((-0.5, 29.5))
    assert ax_image.get_ylim() == pytest.approx((19.5, -0.5))
    plt.close(fig)


@pytest.mark.parametrize("image_px", [None, PIXEL])
def test_profiles_draws_markers_on_their_own_cut(image_px):
    lines = cuts()
    first, second = markers(lines[0], (3, 10)), markers(lines[0], (5, 20))
    fig, (ax_image, ax_profile) = plotting.profiles(
        BIG, lines, pixel_size=image_px, measurements=[second, first])
    # 2 cut lines, then one line of 2 dots per set of markers.
    dots = ax_image.lines[2:]
    assert len(dots) == 2 and len(ax_profile.lines) == 2 + 2 * (1 + 2)
    # Each set of markers has its own colour, not the colour of its cut.
    colors = [drawn.get_color() for drawn in dots]
    assert colors[0] != colors[1] and ax_image.lines[0].get_color() not in colors
    # Cut 1 is horizontal from (col 2, row 3): distance 3 is at column 5.
    x, y = dots[1].get_xydata()[0]
    expected = (5, 3) if image_px is None else (5 * PIXEL[0], (19 - 3) * PIXEL[1])
    assert (x, y) == pytest.approx(expected)
    np.testing.assert_allclose(ax_profile.lines[2].get_xydata(),
                               np.column_stack([second.distance, second.values]))
    plt.close(fig)


def test_profiles_measurement_on_a_cut_not_plotted_raises():
    lines = cuts()
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="not in profiles"):
        plotting.profiles(BIG, lines[1:], measurements=[markers(lines[0], (3, 10))])
    with pytest.raises(ValueError, match="not in profiles"):
        plotting.profiles(BIG, cuts(), measurements=[markers(lines[0], (3, 10))])  # remade
    with pytest.raises(TypeError, match="analysis.markers"):
        plotting.profiles(BIG, lines, measurements=[(3, 10)])
    assert plt.get_fignums() == before


def test_profiles_on_given_axes():
    fig, axes = plt.subplots(1, 2)
    out_fig, out_axes = plotting.profiles(BIG, cuts(), ax=axes)
    assert out_fig is fig and out_axes[0] is axes[0] and out_axes[1] is axes[1]
    plt.close(fig)


def test_profiles_bad_input_raises_and_leaves_no_figure():
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="empty"):
        plotting.profiles(BIG, [])
    with pytest.raises(ValueError, match="m and others in px"):
        plotting.profiles(BIG, cuts() + cuts(PIXEL))
    with pytest.raises(ValueError, match="different image"):
        plotting.profiles(BIG[:5, :5], cuts())
    with pytest.raises(ValueError, match="mask shape"):
        plotting.profiles(BIG, cuts(), mask=np.zeros((2, 2), dtype=bool))
    assert plt.get_fignums() == before


def test_profiles_draws_step_height_ranges_on_their_cut():
    lines = cuts()
    s = step_height(lines[1], a=(0, 5), b=(10, 15))
    fig, (ax_image, ax_profile) = plotting.profiles(BIG, lines, measurements=[s])
    assert len(ax_profile.patches) == 2                 # one shaded span per range
    level_lines = ax_profile.lines[2:]
    for line, (lo, hi), level in zip(level_lines, s.ranges, s.levels):
        np.testing.assert_allclose(line.get_xydata(), [[lo, level], [hi, level]])
        assert line.get_color() == level_lines[0].get_color()      # one colour per result
    # On the image: two thick stretches of cut 2, which starts at (col 4, row 15).
    stretch = ax_image.lines[2].get_xydata()
    np.testing.assert_allclose(stretch[0], lines[1].start_px)
    plt.close(fig)


def test_profiles_draws_crossings_on_their_cut():
    lines = cuts()
    x = crossings(lines[0], 15e-9)                  # cut 1 runs along row 3: 3 + col nm
    fig, (ax_image, ax_profile) = plotting.profiles(BIG, lines, measurements=[x])
    level_line, diamonds = ax_profile.lines[2:]
    np.testing.assert_allclose(level_line.get_ydata(), [15e-9, 15e-9])
    np.testing.assert_allclose(diamonds.get_xdata(), x.distance)
    # On the image the crossing is at column 12, on row 3.
    np.testing.assert_allclose(ax_image.lines[2].get_xydata(), [[12, 3]])
    plt.close(fig)


def test_profiles_legend_has_lines_then_one_entry_per_measurement():
    dx, dy = PIXEL
    lines = cuts(PIXEL)                              # cut 1: row 3, 1 nm per column
    measurements = [
        markers(lines[0], (3 * dx, 10 * dx)),        # 7 columns: 3.5 µm, 7 nm
        # Columns 2-4 (mean 6 nm) and 22-25 (mean 26.5 nm); half-pixel margins so no
        # sample sits on a range end.
        step_height(lines[0], a=(0, 2.5 * dx), b=(19.5 * dx, lines[0].distance[-1])),
        crossings(lines[0], 15e-9),                  # one crossing, at column 12
        crossings(lines[0], 100e-9),                 # none
    ]
    fig, (_, ax_profile) = plotting.profiles(BIG, lines, pixel_size=PIXEL,
                                             measurements=measurements)
    texts = [t.get_text() for t in ax_profile.get_legend().get_texts()]
    assert texts == [
        "line 1",
        "line 2",
        r"$\Delta x$ = 3.5 µm, $\Delta z$ = 7 nm",
        r"$\Delta z$ = 20.5 nm",
        "x = 5 µm at 15 nm",
        "no crossing at 100 nm",
    ]
    plt.close(fig)


def test_profiles_legend_crossings_width_in_px():
    # Ridge rising from column 5 to 10, falling from 20 to 25: 2.5 nm at 7.5 and 22.5.
    ridge = np.clip(np.minimum(np.arange(30.0) - 5, 25 - np.arange(30.0)), 0, 5)
    x = crossings(profile(np.tile(ridge, (4, 1)) * 1e-9, (0, 1), (29, 1)), 2.5e-9)
    fig, (_, ax_profile) = plotting.profiles(np.tile(ridge, (4, 1)) * 1e-9, [x.profile],
                                             measurements=[x])
    assert ax_profile.get_legend().get_texts()[-1].get_text() == \
        r"$\Delta x$ = 15 px at 2.5 nm"
    plt.close(fig)
