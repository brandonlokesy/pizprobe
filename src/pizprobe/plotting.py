# pizprobe/plotting.py
"""
Quick-look plots for AFM images and masks.

Functions
---------
image
    Show an image, optionally with a mask drawn on top.
histogram
    Histogram of the image values, to choose thresholds for ``pizprobe.mask.threshold``.

Conventions
-----------
- Values are plotted in SI units with an SI prefix on the axis (``EngFormatter``): a tick
  reading "12 nm" is the value ``12e-9``. Numbers read off these plots can be passed
  straight to ``pizprobe.mask`` functions.
- Masked points (True) are drawn in blue on top of the image, and in blue in the
  histogram.
- Both functions draw on ``ax`` if given, otherwise on a new figure, and return
  ``(fig, ax)`` so the plot can be adjusted further with matplotlib (same as ``leman``).
  For a chosen size, make the figure first: ``fig, ax = plt.subplots(figsize=(6, 5))``,
  then pass ``ax=ax``. The drawn image is ``ax.images[0]``; the colour bar of
  ``image`` is ``fig.axes[-1]``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import EngFormatter

from .mask import _check_mask, _check_pixel_size

_MASK_COLOR = "tab:blue"
_MASK_ALPHA = 0.5


def image(z, mask=None, pixel_size=None, units="m", ax=None, **imshow_kwargs):
    """
    Show an image, optionally with a mask drawn on top.

    Parameters
    ----------
    z : np.ndarray
        2-D image, rows = scan lines, as in ``Channel.data``.
    mask : np.ndarray of bool, optional
        Same shape as ``z``. True points are drawn in semi-transparent blue.
    pixel_size : tuple of float, optional
        ``(dx, dy)`` in m (e.g. ``scan.pixel_size``). None: axes in pixels, origin top
        left. Given: axes in m, origin at the bottom left, every pixel centre at
        ``(col * dx, (n_rows - 1 - row) * dy)`` -- the same coordinates as
        ``pizprobe.mask.rectangle`` / ``polygon`` with ``pixel_size``.
    units : str
        Units of ``z``, for the colour bar (e.g. ``Channel.units``). Default ``"m"``.
    ax : matplotlib Axes, optional
        Axes to draw on. Default: a new figure.
    **imshow_kwargs
        Passed to ``ax.imshow`` for the image, e.g. ``cmap``, ``vmin``, ``vmax``.
        Defaults: ``cmap="afmhot"``, ``interpolation="nearest"``.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Examples
    --------
    >>> ch = scan["ZSensorRetrace"]
    >>> fig, ax = plotting.image(ch.data, mask=features, pixel_size=scan.pixel_size, units=ch.units)
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError(f"z must be a 2-D image, got shape {z.shape}.")
    mask = _check_mask(mask, z.shape)

    extent = None
    if pixel_size is not None:
        dx, dy = _check_pixel_size(pixel_size)
        n_rows, n_cols = z.shape
        extent = [-dx / 2, (n_cols - 0.5) * dx, -dy / 2, (n_rows - 0.5) * dy]

    fig, ax = _figure_and_axes(ax)

    imshow_kwargs.setdefault("cmap", "afmhot")
    imshow_kwargs.setdefault("interpolation", "nearest")
    shown = ax.imshow(z, extent=extent, **imshow_kwargs)
    fig.colorbar(shown, ax=ax, format=EngFormatter(unit=units))

    if mask is not None:
        overlay = np.ma.masked_where(~mask, np.ones(z.shape))
        ax.imshow(overlay, extent=extent, cmap=ListedColormap([_MASK_COLOR]),
                  alpha=_MASK_ALPHA, interpolation="nearest")

    if extent is None:
        ax.set_xlabel("column (px)")
        ax.set_ylabel("row (px)")
    else:
        ax.xaxis.set_major_formatter(EngFormatter(unit="m"))
        ax.yaxis.set_major_formatter(EngFormatter(unit="m"))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    return fig, ax


def histogram(z, mask=None, bins=256, thresholds=(), units="m", ax=None):
    """
    Histogram of the image values, to choose thresholds for ``pizprobe.mask.threshold``.

    On a levelled image, each flat level (substrate, flake, layer, ridge top) shows up as
    a peak; a threshold between two peaks separates them.

    Parameters
    ----------
    z : np.ndarray
        Image. Non-finite points are left out.
    mask : np.ndarray of bool, optional
        Same shape as ``z``. The masked points (True) are also shown on their own, in blue,
        to check which peaks the mask covers.
    bins : int
        Number of bins over the range of ``z``. Default 256.
    thresholds : sequence of float
        Values drawn as dashed vertical lines, in the units of ``z``.
    units : str
        Units of ``z``, for the horizontal axis. Default ``"m"``.
    ax : matplotlib Axes, optional
        Axes to draw on. Default: a new figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes

    Examples
    --------
    >>> fig, ax = plotting.histogram(z, thresholds=[12e-9])
    >>> ax.set_yscale("log")            # small peaks next to a large substrate peak
    >>> features = mask.threshold(z, above=12e-9)
    """
    z = np.asarray(z, dtype=float)
    mask = _check_mask(mask, z.shape)

    finite = np.isfinite(z)
    if not finite.any():
        raise ValueError("z has no finite values to histogram.")
    edges = np.histogram_bin_edges(z[finite], bins=bins)
    fig, ax = _figure_and_axes(ax)

    ax.hist(z[finite], bins=edges, histtype="stepfilled", color="0.7", label="all points")
    if mask is not None:
        ax.hist(z[finite & mask], bins=edges, histtype="step", color=_MASK_COLOR,
                label="masked")
    for value in thresholds:
        ax.axvline(value, color="k", linestyle="--", linewidth=1)

    ax.xaxis.set_major_formatter(EngFormatter(unit=units))
    ax.set_xlabel("value")
    ax.set_ylabel("points")
    if mask is not None:
        ax.legend()
    return fig, ax


def _figure_and_axes(ax):
    """``(fig, ax)``: a new figure if ``ax`` is None, else ``ax`` and its figure."""
    import matplotlib.pyplot as plt

    if ax is None:
        return plt.subplots()
    return ax.get_figure(), ax
