"""plotting.image and plotting.histogram, drawn off-screen."""

import matplotlib

matplotlib.use("Agg")  # no window; must come before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pizprobe import plotting
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
