# pizprobe/mask.py
"""
Boolean masks for AFM images.

Functions
---------
threshold
    Mask of the points above and/or below given heights.
rectangle
    Mask of the pixels inside an axis-aligned rectangle.
polygon
    Mask of the pixels inside a polygon.
grow, shrink
    Move the edge of a mask outwards or inwards by a number of pixels.
remove_small
    Drop isolated patches smaller than a number of pixels.

Conventions
-----------
- A mask is a boolean array with the shape of the image. In the fitting functions of
  ``pizprobe.processing``, **True marks a feature**: those points are left out of the fit.
  This is the same meaning as a mask in the Asylum software, TopoStats and
  ``numpy.ma``.
- Region functions return True **inside** the shape. Whether the region is a feature or
  the only area to fit is decided by how it is combined:

  ============================================  ==================================
  Asylum Modify Panel                           PizProbe (``features``: True = left out)
  ============================================  ==================================
  Exclude Points (add a region to the mask)     ``features = features | region``
  Include Points (remove a region from mask)    ``features = features & ~region``
  Invert                                        ``features = ~features``
  Fit only inside a box (guide 7.2.1.2)         ``features = ~region``
  ============================================  ==================================

- Coordinates are ``(horizontal, vertical)``, **as read off the plot you are looking at**:

  - ``pixel_size=None``: pixels. Origin at the top left, vertical axis down, as in
    ``plt.imshow(z)`` and ``z[row, col]``.
  - ``pixel_size=(dx, dy)`` in m (e.g. ``scan.pixel_size``): metres. Origin at the first
    point of the first scan line (bottom left of the displayed image), vertical axis up.
    ``x = col * dx`` and ``y = (n_rows - 1 - row) * dy``. ``pizprobe.plotting.image(z,
    pixel_size=scan.pixel_size)`` draws the image with exactly these axes.

  A pixel belongs to a region when its centre lies inside the shape.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy import ndimage

# Pixel-coordinate tolerance, so a boundary given in metres that lands exactly on a pixel
# centre (up to rounding in x / dx) includes that pixel.
_EDGE = 1e-9


def threshold(z: np.ndarray, above: float | None = None, below: float | None = None) -> np.ndarray:
    """
    Mask of the points above and/or below given heights.

    Pick the values from ``pizprobe.plotting.histogram``. Combine masks with ``|``, ``&`` and
    ``~`` for other selections (see the module docstring).

    Parameters
    ----------
    z : np.ndarray
        2-D image.
    above : float, optional
        True where ``z > above``. In the units of ``z`` (m for heights).
    below : float, optional
        True where ``z < below``. Given together with ``above``, the mask is the band
        ``above < z < below``.

    Returns
    -------
    np.ndarray of bool
        Same shape as ``z``. Non-finite points are False.

    Raises
    ------
    ValueError
        If neither limit is given, or ``above >= below`` (the band would be empty).

    Examples
    --------
    >>> features = threshold(z, above=12e-9)                  # particles on a substrate
    >>> layer = threshold(z, above=0.5e-9, below=1.2e-9)      # one flake layer
    >>> features = threshold(z, above=12e-9) | threshold(z, below=-3e-9)  # bumps and pits
    """
    if above is None and below is None:
        raise ValueError("Give at least one of above= or below=.")
    if above is not None and below is not None and above >= below:
        raise ValueError(
            f"above={above:g} is not smaller than below={below:g}, so the band "
            "above < z < below is empty. For points outside a band, use "
            "threshold(z, above=high) | threshold(z, below=low)."
        )
    z = np.asarray(z, dtype=float)
    out = np.isfinite(z)
    if above is not None:
        out &= z > above
    if below is not None:
        out &= z < below
    return out


def rectangle(shape, x, y, pixel_size=None) -> np.ndarray:
    """
    Mask of the pixels inside an axis-aligned rectangle.

    Parameters
    ----------
    shape : tuple of int
        ``(n_rows, n_cols)`` of the image, e.g. ``z.shape``.
    x, y : tuple of float
        ``(start, end)`` of the rectangle along the horizontal and vertical axes, in
        either order. Both ends are included.
    pixel_size : tuple of float, optional
        ``(dx, dy)`` in m. None: ``x`` and ``y`` are pixels (origin top left). Given:
        ``x`` and ``y`` are metres (origin bottom left). See the module docstring.

    Returns
    -------
    np.ndarray of bool
        True inside the rectangle.

    Examples
    --------
    >>> region = rectangle(z.shape, x=(300, 511), y=(0, 511))       # pixels
    >>> region = rectangle(z.shape, x=(6e-6, 8e-6), y=(0, 20e-6),
    ...                    pixel_size=scan.pixel_size)             # metres
    >>> features = ~region                                          # fit only inside
    """
    shape = _check_shape(shape)
    corners = _to_pixels([(x[0], y[0]), (x[1], y[1])], shape, pixel_size)
    (c0, c1), (r0, r1) = np.sort(corners[:, 0]), np.sort(corners[:, 1])
    cols = np.arange(shape[1])
    rows = np.arange(shape[0])
    in_cols = (cols >= c0 - _EDGE) & (cols <= c1 + _EDGE)
    in_rows = (rows >= r0 - _EDGE) & (rows <= r1 + _EDGE)
    return in_rows[:, None] & in_cols[None, :]


def polygon(shape, vertices, pixel_size=None) -> np.ndarray:
    """
    Mask of the pixels inside a polygon.

    Parameters
    ----------
    shape : tuple of int
        ``(n_rows, n_cols)`` of the image, e.g. ``z.shape``.
    vertices : sequence of (float, float)
        At least 3 ``(horizontal, vertical)`` corners, in order around the polygon. The
        polygon is closed automatically.
    pixel_size : tuple of float, optional
        ``(dx, dy)`` in m. None: vertices are pixels (origin top left). Given: vertices
        are metres (origin bottom left). See the module docstring.

    Returns
    -------
    np.ndarray of bool
        True for pixels whose centre is inside the polygon. Pixels whose centre lies
        exactly on an edge may fall either way.

    Examples
    --------
    >>> region = polygon(z.shape, [(10, 10), (200, 40), (120, 300)])   # pixels
    >>> features = features | region                                   # exclude it
    """
    shape = _check_shape(shape)
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError(
            "vertices must be at least 3 (horizontal, vertical) pairs, got an array of "
            f"shape {vertices.shape}."
        )
    path = Path(_to_pixels(vertices, shape, pixel_size))
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    centres = np.column_stack([cols.ravel(), rows.ravel()])
    return path.contains_points(centres).reshape(shape)


def grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    """
    Move the edge of a mask outwards by ``pixels``.

    Every pixel within a distance of ``pixels`` (a disc, measured between pixel centres)
    of a True pixel becomes True. Use it to keep the foot of a step or the rim of a
    particle out of a fit. The Asylum software calls this "Dilate Mask".

    Parameters
    ----------
    mask : np.ndarray of bool
        2-D mask.
    pixels : int
        Distance in pixels, >= 0. 0 returns a copy.

    Returns
    -------
    np.ndarray of bool

    Examples
    --------
    >>> features = grow(threshold(z, above=15e-9), 3)
    """
    mask, pixels = _check_morphology(mask, pixels)
    if pixels == 0:
        return mask.copy()
    return ndimage.binary_dilation(mask, structure=_disc(pixels))


def shrink(mask: np.ndarray, pixels: int) -> np.ndarray:
    """
    Move the edge of a mask inwards by ``pixels``.

    A pixel stays True only if every pixel within a distance of ``pixels`` is True, so
    patches narrower than about ``2 * pixels + 1`` disappear. The image border does not
    count as an edge: a patch touching the border shrinks only from its edges inside the
    image. The Asylum software calls this "Erode Mask". ``grow`` then ``shrink`` by the
    same amount is not always the identity (it fills narrow gaps).

    Parameters
    ----------
    mask : np.ndarray of bool
        2-D mask.
    pixels : int
        Distance in pixels, >= 0. 0 returns a copy.

    Returns
    -------
    np.ndarray of bool
    """
    mask, pixels = _check_morphology(mask, pixels)
    if pixels == 0:
        return mask.copy()
    return ndimage.binary_erosion(mask, structure=_disc(pixels), border_value=1)


def remove_small(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """
    Drop isolated patches of True smaller than ``min_pixels``.

    Patches are groups of True pixels touching by an edge or a corner. Use it to clean
    single noisy pixels out of a threshold mask. To fill small holes instead, apply it
    to the inverted mask: ``~remove_small(~mask, n)``.

    Parameters
    ----------
    mask : np.ndarray of bool
        2-D mask.
    min_pixels : int
        Patches with fewer pixels than this are set to False. >= 1.

    Returns
    -------
    np.ndarray of bool

    Examples
    --------
    >>> features = remove_small(threshold(z, above=15e-9), 5)    # drop noise specks
    >>> features = ~remove_small(~features, 20)                  # fill small holes
    """
    mask, min_pixels = _check_morphology(mask, min_pixels, name="min_pixels", minimum=1)
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False  # label 0 is the background
    return keep[labels]


def _disc(radius: int) -> np.ndarray:
    """Boolean disc of ``radius`` pixels, centre included."""
    r = np.arange(-radius, radius + 1)
    return r[:, None] ** 2 + r[None, :] ** 2 <= radius**2


def _check_morphology(mask, n, name="pixels", minimum=0):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}.")
    if int(n) != n or n < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {n!r}.")
    return mask, int(n)


def _check_shape(shape) -> tuple:
    shape = tuple(int(n) for n in shape)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"shape must be (n_rows, n_cols) of a 2-D image, got {shape}.")
    return shape


def _check_mask(mask, shape):
    """``mask`` as a bool array of ``shape``, or None. Shared by every function taking a mask."""
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError(f"mask shape {mask.shape} does not match z shape {tuple(shape)}.")
    return mask


def _check_pixel_size(pixel_size) -> tuple:
    """``pixel_size`` as ``(dx, dy)`` floats, both > 0. Shared with ``pizprobe.plotting``."""
    dx, dy = (float(s) for s in pixel_size)
    if not (dx > 0 and dy > 0):
        raise ValueError(f"pixel_size must be two positive lengths in m, got {pixel_size}.")
    return dx, dy


def _to_pixels(points, shape, pixel_size) -> np.ndarray:
    """``(horizontal, vertical)`` points -> ``(col, row)`` pixel coordinates, as floats."""
    points = np.asarray(points, dtype=float)
    if pixel_size is None:
        return points
    dx, dy = _check_pixel_size(pixel_size)
    return np.column_stack([points[:, 0] / dx, (shape[0] - 1) - points[:, 1] / dy])
