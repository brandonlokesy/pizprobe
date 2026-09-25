# pizprobe/mask.py
"""
Boolean masks for AFM images.

Functions
---------
rectangle
    Mask of the pixels inside an axis-aligned rectangle.
polygon
    Mask of the pixels inside a polygon.

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
    ``x = col * dx`` and ``y = (n_rows - 1 - row) * dy``. This matches
    ``plt.imshow(z, extent=[-dx/2, sx + dx/2, -dy/2, sy + dy/2])`` with
    ``sx, sy = scan.size``, which puts every pixel centre at its true position.

  A pixel belongs to a region when its centre lies inside the shape.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path

# Pixel-coordinate tolerance, so a boundary given in metres that lands exactly on a pixel
# centre (up to rounding in x / dx) includes that pixel.
_EDGE = 1e-9


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


def _check_shape(shape) -> tuple:
    shape = tuple(int(n) for n in shape)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"shape must be (n_rows, n_cols) of a 2-D image, got {shape}.")
    return shape


def _to_pixels(points, shape, pixel_size) -> np.ndarray:
    """``(horizontal, vertical)`` points -> ``(col, row)`` pixel coordinates, as floats."""
    points = np.asarray(points, dtype=float)
    if pixel_size is None:
        return points
    dx, dy = (float(s) for s in pixel_size)
    if not (dx > 0 and dy > 0):
        raise ValueError(f"pixel_size must be two positive lengths in m, got {pixel_size}.")
    return np.column_stack([points[:, 0] / dx, (shape[0] - 1) - points[:, 1] / dy])
