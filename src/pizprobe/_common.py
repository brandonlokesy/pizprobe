# pizprobe/_common.py
"""
Input checks and coordinate conversion shared by several modules.

Private to the package (leading underscore): not part of the public interface, so names
and signatures here may change without notice. Each check raises ``ValueError`` with a
message meant for the user of the public function that called it.
"""

from __future__ import annotations

import numpy as np


def _as_image(z) -> np.ndarray:
    """``z`` as a 2-D float array."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"z must be a 2-D image, got shape {z.shape}.")
    return z


def _check_shape(shape) -> tuple:
    """``shape`` as ``(n_rows, n_cols)`` ints, both >= 1."""
    shape = tuple(int(n) for n in shape)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"shape must be (n_rows, n_cols) of a 2-D image, got {shape}.")
    return shape


def _check_mask(mask, shape):
    """``mask`` as a bool array of ``shape``, or None."""
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError(f"mask shape {mask.shape} does not match z shape {tuple(shape)}.")
    return mask


def _check_pixel_size(pixel_size) -> tuple:
    """``pixel_size`` as ``(dx, dy)`` floats, both > 0."""
    dx, dy = (float(s) for s in pixel_size)
    if not (dx > 0 and dy > 0):
        raise ValueError(f"pixel_size must be two positive lengths in m, got {pixel_size}.")
    return dx, dy


def _to_pixels(points, shape, pixel_size) -> np.ndarray:
    """
    ``(horizontal, vertical)`` points -> ``(col, row)`` pixel coordinates, as floats.

    ``pixel_size=None``: points are already pixels. Otherwise points are metres, origin at
    the bottom left: ``col = x / dx``, ``row = (n_rows - 1) - y / dy``.
    """
    points = np.asarray(points, dtype=float)
    if pixel_size is None:
        return points
    dx, dy = _check_pixel_size(pixel_size)
    return np.column_stack([points[:, 0] / dx, (shape[0] - 1) - points[:, 1] / dy])


def _from_pixels(points, shape, pixel_size) -> np.ndarray:
    """``(col, row)`` pixel coordinates -> ``(horizontal, vertical)``; inverse of ``_to_pixels``."""
    points = np.asarray(points, dtype=float)
    if pixel_size is None:
        return points
    dx, dy = _check_pixel_size(pixel_size)
    return np.column_stack([points[:, 0] * dx, ((shape[0] - 1) - points[:, 1]) * dy])
