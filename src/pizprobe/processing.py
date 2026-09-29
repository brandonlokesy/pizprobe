# pizprobe/processing.py
"""
Levelling corrections for AFM height images.

Functions
---------
plane_fit
    Fit a 2-D polynomial surface (plane, bowl, ...) to the unmasked points and subtract it.
line_flatten
    Fit each scan line on its own (offset, or offset + slope) and subtract it.

Conventions
-----------
- ``z`` is a 2-D image with rows = scan lines, as in ``Channel.data``.
- ``mask`` is a boolean array of the same shape. **True marks a feature**: masked points
  are left out of the fit, but the fitted surface is still subtracted from them.
- Every function returns a new array; the input is not modified.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._common import _as_image, _check_mask


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
    z = _as_image(z)
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


def line_flatten(
    z: np.ndarray,
    mask: np.ndarray | None = None,
    order: int = 0,
    statistic: str = "mean",
) -> np.ndarray:
    """
    Fit each scan line on its own and subtract the fit.

    Removes line-to-line offsets (order 0) or offsets and slopes (order 1), e.g. from
    Z drift or tip changes between lines. Each line is fitted on its unmasked, finite
    points: order 0 subtracts their mean (or median), order 1 the least-squares straight
    line through them. Any real height change from one line to the next is removed as well,
    so use it only when the image shows line offsets.

    An order-1 flatten also removes any plane, so ``plane_fit(z, order=1)`` before it
    does not change its result. An order-0 flatten leaves the slope along the lines.

    Parameters
    ----------
    z : np.ndarray
        2-D image, rows = scan lines.
    mask : np.ndarray of bool, optional
        Same shape as ``z``. True = feature, left out of the fit. None fits every point.
    order : {0, 1}
        0: offset per line. 1: offset and slope per line.
    statistic : {"mean", "median"}
        Order 0 only: the offset subtracted from each line. ``"mean"`` (default) is the
        least-squares offset, as in the Asylum software ("all of the lines have the same
        average value"). ``"median"`` is less pulled by unmasked bumps or residue on the
        floor, as in TopoStats' median flatten.

    Returns
    -------
    np.ndarray
        ``z`` with each line's fit subtracted, as float. A line with fewer than
        ``order + 1`` unmasked points is returned unchanged, with a ``UserWarning``
        giving the number of such lines.

    Raises
    ------
    ValueError
        If ``z`` is not 2-D, ``mask`` has the wrong shape, ``order`` is not 0 or 1,
        ``statistic`` is unknown, or ``statistic="median"`` is given with ``order=1``.

    Examples
    --------
    >>> flat = line_flatten(levelled, mask=features)            # offsets only
    >>> flat = line_flatten(levelled, mask=features, statistic="median")
    >>> flat = line_flatten(levelled, mask=features, order=1)   # offsets and slopes
    """
    z = _as_image(z)
    if order not in (0, 1):
        raise ValueError(f"order must be 0 or 1, got {order!r}.")
    if statistic not in ("mean", "median"):
        raise ValueError(f'statistic must be "mean" or "median", got {statistic!r}.')
    if statistic == "median" and order != 0:
        raise ValueError(
            'statistic="median" is only defined for order=0 (an offset per line). '
            "Use order=0, or the default least-squares fit for order=1."
        )

    use = np.isfinite(z)
    mask = _check_mask(mask, z.shape)
    if mask is not None:
        use &= ~mask

    # Per-line weighted least squares in closed form, with w = 1 on used points, else 0.
    w = use.astype(float)
    zw = np.where(use, z, 0.0)
    n = w.sum(axis=1)
    fitted = n >= order + 1
    safe_n = np.where(fitted, n, 1.0)

    if order == 0 and statistic == "median":
        offset = np.zeros(z.shape[0])
        offset[fitted] = np.nanmedian(np.where(use, z, np.nan)[fitted], axis=1)
        line = offset[:, None]
    elif order == 0:
        line = (zw.sum(axis=1) / safe_n)[:, None]
    else:
        u = np.linspace(-1, 1, z.shape[1])
        su, suu = w @ u, w @ u**2
        sz, suz = zw.sum(axis=1), zw @ u
        det = np.where(fitted, n * suu - su**2, 1.0)
        slope = (n * suz - su * sz) / det
        offset = (sz - slope * su) / safe_n
        line = offset[:, None] + slope[:, None] * u[None, :]

    skipped = int(np.sum(~fitted))
    if skipped:
        warnings.warn(
            f"{skipped} of {z.shape[0]} scan lines have fewer than {order + 1} unmasked "
            "points and were left uncorrected. Mask fewer points or lower the order.",
            UserWarning,
            stacklevel=2,
        )
    return np.where(fitted[:, None], z - line, z)
