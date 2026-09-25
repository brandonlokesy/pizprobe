# pizprobe/processing.py
"""
Levelling corrections for AFM height images.

Functions
---------
plane_fit
    Fit a 2-D polynomial surface (plane, bowl, ...) to the unmasked points and subtract it.

Conventions
-----------
- ``z`` is a 2-D image with rows = scan lines, as in ``Channel.data``.
- ``mask`` is a boolean array of the same shape. **True marks a feature**: masked points
  are left out of the fit, but the fitted surface is still subtracted from them.
- Every function returns a new array; the input is not modified.
"""

from __future__ import annotations

import numpy as np

from .mask import _check_mask


def plane_fit(z: np.ndarray, mask: np.ndarray | None = None, order: int = 1) -> np.ndarray:
    """
    Subtract a 2-D polynomial surface fitted to the unmasked points.

    The surface is ``sum(c_pq * u**p * v**q)`` over all ``p + q <= order``, with the column
    and row coordinates ``u``, ``v`` scaled to [-1, 1]. Order 1 is a plane (sample tilt),
    order 2 a bowl (scanner bow). Non-finite points are left out of the fit.

    Parameters
    ----------
    z : np.ndarray
        2-D image.
    mask : np.ndarray of bool, optional
        Same shape as ``z``. True = feature, left out of the fit. None fits every point.
    order : int
        Polynomial order of the surface, >= 0. Order 0 subtracts the mean of the unmasked
        points.

    Returns
    -------
    np.ndarray
        ``z`` minus the fitted surface, as float.

    Raises
    ------
    ValueError
        If ``z`` is not 2-D, ``mask`` has the wrong shape, ``order`` is negative, or the
        unmasked points cannot define the surface (too few, or all on one line).

    Examples
    --------
    >>> levelled = plane_fit(z)                     # rough level, no mask
    >>> levelled = plane_fit(z, mask=features)      # fit the floor only
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"z must be a 2-D image, got shape {z.shape}.")
    if int(order) != order or order < 0:
        raise ValueError(f"order must be an integer >= 0, got {order!r}.")
    order = int(order)

    use = np.isfinite(z)
    mask = _check_mask(mask, z.shape)
    if mask is not None:
        use &= ~mask

    terms = [(p, q) for p in range(order + 1) for q in range(order + 1 - p)]
    rows, cols = np.nonzero(use)
    if len(rows) < len(terms):
        raise ValueError(
            f"Only {len(rows)} unmasked points; an order-{order} surface needs at least "
            f"{len(terms)}. Mask fewer points or lower the order."
        )

    u = np.linspace(-1, 1, z.shape[1])
    v = np.linspace(-1, 1, z.shape[0])
    design = np.column_stack([u[cols] ** p * v[rows] ** q for p, q in terms])
    coeffs, _, rank, _ = np.linalg.lstsq(design, z[rows, cols], rcond=None)
    if rank < len(terms):
        raise ValueError(
            f"The unmasked points cannot define an order-{order} surface (for example, "
            "they all lie in one row or column). Mask fewer points or lower the order."
        )

    surface = sum(c * u[None, :] ** p * v[:, None] ** q for c, (p, q) in zip(coeffs, terms))
    return z - surface
