# pizprobe/plotting.py
"""
Quick-look plots for AFM images and masks.

Functions
---------
image
    Show an image, optionally with a mask drawn on top.
histogram
    Histogram of the image values, to choose thresholds for ``pizprobe.mask.threshold``.
profiles
    Line cuts drawn on the image (left) and their profiles (right), in matching colours.

Conventions
-----------
- Values are plotted in SI units. In ``histogram`` the SI prefix is on each tick
  (``EngFormatter``): a tick reading "12 nm" is the value ``12e-9``, and can be passed
  straight to ``pizprobe.mask`` functions. In ``image`` and ``profiles`` one prefix is
  chosen per axis and shown in the axis label, e.g. "x (µm)": a tick reading "2.5" there
  is the value ``2.5e-6`` for ``pizprobe.mask`` and ``pizprobe.analysis``.
- Masked points (True) are drawn in blue on top of the image, and in blue in the
  histogram.
- Every function draws on ``ax`` if given, otherwise on a new figure, and returns
  ``(fig, ax)`` (``profiles``: ``(fig, (ax_image, ax_profile))``) so the plot can be adjusted further with matplotlib (same as ``leman``).
  For a chosen size, make the figure first: ``fig, ax = plt.subplots(figsize=(6, 5))``,
  then pass ``ax=ax``. The drawn image is ``ax.images[0]``; the colour bar of
  ``image`` is ``fig.axes[-1]``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import EngFormatter, FuncFormatter

from ._common import _as_image, _check_mask, _check_pixel_size, _from_pixels
from .analysis import Crossings, Markers, StepHeight

_MASK_COLOR = "tab:blue"
_MASK_ALPHA = 0.5
# Fixed hex codes, so a matplotlib style or seaborn theme cannot change them. Both lists
# leave out the blue (close to the mask colour) and the grey (hard to see on "gray").
# Line cuts: light, from ColorBrewer Set2 (matplotlib "Set2").
_LINE_COLORS = ["#66c2a5", "#fc8d62", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494"]
# Measurements: dark, from ColorBrewer Dark2 (matplotlib "Dark2"), the same hues as Set2
# but darker. The order is shifted so measurement 1 is not the hue of line 1.
_MARKER_COLORS = ["#e7298a", "#7570b3", "#d95f02", "#1b9e77", "#66a61e", "#e6ab02",
                  "#a6761d"]
_BAND_ALPHA = 0.3


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
        left. Given: axes in m with one SI prefix in the axis labels (e.g. "x (µm)"),
        origin at the bottom left, every pixel centre at
        ``(col * dx, (n_rows - 1 - row) * dy)`` -- the same coordinates as
        ``pizprobe.mask.rectangle`` / ``polygon`` with ``pixel_size``.
    units : str
        Units of ``z`` (e.g. ``Channel.units``), shown with an SI prefix as the colour-bar
        label. Default ``"m"``.
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
    z = _as_image(z)
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
    scale, label = _si_scale(np.max(np.abs(shown.get_clim())), units)
    colorbar = fig.colorbar(shown, ax=ax, format=_scaled(scale))
    colorbar.set_label(label)

    if mask is not None:
        overlay = np.ma.masked_where(~mask, np.ones(z.shape))
        ax.imshow(overlay, extent=extent, cmap=ListedColormap([_MASK_COLOR]),
                  alpha=_MASK_ALPHA, interpolation="nearest")

    if extent is None:
        ax.set_xlabel("column (px)")
        ax.set_ylabel("row (px)")
    else:
        scale, label = _si_scale(max(extent[1], extent[3]), "m")   # same prefix on x and y
        ax.xaxis.set_major_formatter(_scaled(scale))
        ax.yaxis.set_major_formatter(_scaled(scale))
        ax.set_xlabel(f"x ({label})")
        ax.set_ylabel(f"y ({label})")
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


def profiles(z, profiles, pixel_size=None, units="m", mask=None, measurements=(), ax=None,
             **imshow_kwargs):
    """
    Line cuts drawn on the image (left) and their profiles (right), in matching colours.

    A profile averaged over a band (``width > 1`` in ``analysis.profile``) is also drawn
    as a semi-transparent strip ``width`` px wide around its line: the pixels that went
    into the mean.

    Each measurement gets its own colour, from a darker palette than the lines, and its
    own legend entry with its result:

    - ``analysis.markers``: dots with dashed vertical lines on the profile, dots on the
      image. Legend: Δx and Δz.
    - ``analysis.step_height``: each range as a shaded span with a line at its mean
      level on the profile, and as a thick stretch of the cut line on the image.
      Legend: Δz.
    - ``analysis.crossings``: the level as a dotted line with a diamond at each crossing
      on the profile, diamonds on the image. Legend: Δx for two crossings, else the
      position of each.

    The legend can be changed with matplotlib on the returned ``ax_profile``; only the
    lines and the measurements carry labels, so ``ax_profile.legend(...)`` rebuilds it::

        ax_profile.get_legend().remove()                              # no legend
        ax_profile.legend(handles=ax_profile.get_legend_handles_labels()[0][:2])  # lines only
        ax_profile.legend(loc="upper left", bbox_to_anchor=(1, 1))    # outside the plot

    Parameters
    ----------
    z : np.ndarray
        2-D image the profiles were taken from.
    profiles : sequence of pizprobe.analysis.Profile
        One or more line cuts, e.g. ``[analysis.profile(z, s, e, ...) for s, e in lines]``.
        All must use the same distance unit (all with or all without ``pixel_size``).
    pixel_size : tuple of float, optional
        ``(dx, dy)`` in m, for the image axes (see ``image``). The lines are placed
        correctly whether the profiles were made in pixels or in metres.
    units : str
        Units of ``z`` and of the profile values. Default ``"m"``.
    mask : np.ndarray of bool, optional
        Drawn on the image as in ``image``.
    measurements : sequence of analysis.Markers, StepHeight or Crossings, optional
        Results made on some of the ``profiles``, in any order. Each one is drawn on the
        profile it was made from; a profile can have several, or none.
    ax : pair of matplotlib Axes, optional
        ``(ax_image, ax_profile)`` to draw on. Default: a new figure with two panels.
    **imshow_kwargs
        Passed to ``image`` for the left panel, e.g. ``cmap``, ``vmin``, ``vmax``.

    Returns
    -------
    fig, (ax_image, ax_profile)

    Raises
    ------
    ValueError
        If ``profiles`` is empty, mixes m and px, does not fit on ``z``, or a measurement
        was made on a profile that is not in ``profiles``.
    TypeError
        If a measurement is not an ``analysis.Markers``, ``StepHeight`` or ``Crossings``.

    Examples
    --------
    >>> cuts = [analysis.profile(flat, s, e, pixel_size=px, width=5) for s, e in lines]
    >>> fig, (ax_image, ax_profile) = plotting.profiles(flat, cuts, pixel_size=px)
    >>> m = analysis.markers(cuts[0], (5e-6, 15e-6))
    >>> plotting.profiles(flat, cuts, pixel_size=px, measurements=[m])
    """
    import matplotlib.pyplot as plt

    z = _as_image(z)
    _check_mask(mask, z.shape)          # checked here too, so a bad input leaves no figure
    if pixel_size is not None:
        _check_pixel_size(pixel_size)
    profiles = list(profiles)
    if not profiles:
        raise ValueError("profiles is empty: give at least one analysis.Profile.")
    in_metres = {p.pixel_size is not None for p in profiles}
    if len(in_metres) > 1:
        raise ValueError(
            "Some profiles have distances in m and others in px. Make them all with, or "
            "all without, pixel_size."
        )
    in_metres = in_metres.pop()
    n_rows, n_cols = z.shape
    for i, p in enumerate(profiles, start=1):
        for c, r in (p.start_px, p.end_px):
            if not (-0.5 <= c <= n_cols - 0.5 and -0.5 <= r <= n_rows - 0.5):
                raise ValueError(
                    f"Profile {i} does not fit on this image ({n_cols} x {n_rows} px): "
                    "was it taken from a different image?"
                )
    measurements = list(measurements)
    for j, m in enumerate(measurements, start=1):
        if type(m) not in _DRAW:
            raise TypeError(
                f"Measurement {j} is a {type(m).__name__}; give results of "
                "analysis.markers, analysis.step_height or analysis.crossings."
            )
        if not any(p is m.profile for p in profiles):
            raise ValueError(
                f"Measurement {j} was made on a profile that is not in profiles. Plot "
                "that profile too, or make the measurement again on one that is plotted."
            )

    if ax is None:
        fig, (ax_image, ax_profile) = plt.subplots(1, 2, figsize=(12, 5),
                                                   layout="constrained")
    else:
        ax_image, ax_profile = ax
        fig = ax_image.get_figure()

    image(z, mask=mask, pixel_size=pixel_size, units=units, ax=ax_image, **imshow_kwargs)
    limits = ax_image.get_xlim(), ax_image.get_ylim()
    for i, p in enumerate(profiles):
        color = _LINE_COLORS[i % len(_LINE_COLORS)]
        if p.width > 1:
            ax_image.fill(*_band(p, z.shape, pixel_size).T, color=color,
                          alpha=_BAND_ALPHA, linewidth=0)
        (x0, y0), (x1, y1) = _from_pixels([p.start_px, p.end_px], z.shape, pixel_size)
        ax_image.plot([x0, x1], [y0, y1], color=color, linewidth=1.5)
        ax_image.annotate(str(i + 1), (x0, y0), color=color, fontweight="bold",
                          xytext=(-4, 4), textcoords="offset points", ha="right")
        ax_profile.plot(p.distance, p.values, color=color, label=f"line {i + 1}")
    for j, m in enumerate(measurements):
        _DRAW[type(m)](m, _MARKER_COLORS[j % len(_MARKER_COLORS)], units, ax_image,
                       ax_profile, z.shape, pixel_size)
    ax_image.set_xlim(limits[0])            # a band reaching past the edge must not
    ax_image.set_ylim(limits[1])            # widen the image axes

    values = np.concatenate([p.values for p in profiles])
    values = values[np.isfinite(values)]
    scale, label = _si_scale(np.max(np.abs(values)) if values.size else 0.0, units)
    ax_profile.yaxis.set_major_formatter(_scaled(scale))
    ax_profile.set_ylabel(f"z ({label})")
    if in_metres:
        scale, label = _si_scale(max(p.distance[-1] for p in profiles), "m")
        ax_profile.xaxis.set_major_formatter(_scaled(scale))
        ax_profile.set_xlabel(f"distance ({label})")
    else:
        ax_profile.set_xlabel("distance (px)")
    ax_profile.legend()
    return fig, (ax_image, ax_profile)


# Each _draw_* function draws one measurement in ``color`` and puts its result in the
# label of one artist only, so it has one legend entry; the others are unlabelled.

def _draw_markers(m, color, units, ax_image, ax_profile, shape, pixel_size):
    """Markers ``m`` as dots and dashed lines on the profile, and as dots on the image."""
    style = dict(color=color, marker="o", markeredgecolor="white", linestyle="none")
    label = (rf"$\Delta x$ = {_quantity(m.dx, _distance_unit(m.profile))}, "
             rf"$\Delta z$ = {_quantity(m.dz, units)}")
    ax_profile.plot(m.distance, m.values, label=label, **style)
    for d in m.distance:
        ax_profile.axvline(d, color=color, linestyle="--", linewidth=1)
    points = _on_image(m.profile, m.distance, shape, pixel_size)
    ax_image.plot(points[:, 0], points[:, 1], **style)


def _draw_step_height(s, color, units, ax_image, ax_profile, shape, pixel_size):
    """
    Step height ``s``: each range as a shaded span with its mean level on the profile,
    and as a thick stretch of the cut line on the image.
    """
    label = rf"$\Delta z$ = {_quantity(s.dz, units)}"
    for k, ((lo, hi), level) in enumerate(zip(s.ranges, s.levels)):
        ax_profile.axvspan(lo, hi, color=color, alpha=_BAND_ALPHA, linewidth=0)
        ax_profile.plot([lo, hi], [level, level], color=color, linewidth=2.5,
                        label=label if k == 0 else None)
        ends = _on_image(s.profile, np.array([lo, hi]), shape, pixel_size)
        ax_image.plot(ends[:, 0], ends[:, 1], color=color, linewidth=5, alpha=0.7,
                      solid_capstyle="butt")


def _draw_crossings(x, color, units, ax_image, ax_profile, shape, pixel_size):
    """
    Crossings ``x``: the level as a dotted line along the whole profile with a diamond at
    each crossing, and diamonds on the image.
    """
    p = x.profile
    unit = _distance_unit(p)
    at = f"at {_quantity(x.level, units)}"
    if len(x.distance) == 2:
        label = rf"$\Delta x$ = {_quantity(x.distance[1] - x.distance[0], unit)} {at}"
    elif len(x.distance):
        label = f"x = {', '.join(_quantity(d, unit) for d in x.distance)} {at}"
    else:
        label = f"no crossing {at}"
    ax_profile.plot([0, p.distance[-1]], [x.level, x.level], color=color, linestyle=":",
                    linewidth=1)
    style = dict(color=color, marker="D", markeredgecolor="white", linestyle="none")
    ax_profile.plot(x.distance, np.full(len(x.distance), x.level), label=label, **style)
    points = _on_image(p, x.distance, shape, pixel_size)
    ax_image.plot(points[:, 0], points[:, 1], **style)


def _distance_unit(p):
    """Unit of ``p.distance``: "m" if the profile was made with ``pixel_size``, else "px"."""
    return "m" if p.pixel_size is not None else "px"


def _quantity(value, unit):
    """``value`` with 4 significant figures and an SI prefix: -2.716e-7, "m" -> "-271.6 nm"."""
    if unit == "px":
        return f"{value:.4g} px"
    scale, label = _si_scale(abs(value), unit)
    return f"{value / scale:.4g} {label}"


# How each kind of measurement accepted by ``profiles`` is drawn.
_DRAW = {Markers: _draw_markers, StepHeight: _draw_step_height, Crossings: _draw_crossings}


def _on_image(p, distances, shape, pixel_size):
    """Points at ``distances`` along profile ``p``, in the image-axes coordinates."""
    along = distances / p.distance[-1]              # 0 at start, 1 at end
    start, end = np.array(p.start_px), np.array(p.end_px)
    return _from_pixels(start + along[:, None] * (end - start), shape, pixel_size)


def _band(p, shape, pixel_size):
    """
    Corners of the strip averaged by profile ``p``, in the image-axes coordinates.

    ``analysis.profile`` averages ``width`` lines 1 px apart, centred on the line, so the
    strip is ``width`` px wide (each line stands for 1 px) and runs from start to end.
    """
    (c0, r0), (c1, r1) = p.start_px, p.end_px
    length = np.hypot(c1 - c0, r1 - r0)
    half_c, half_r = (p.width / 2) * np.array([-(r1 - r0), c1 - c0]) / length
    corners = [(c0 + half_c, r0 + half_r), (c1 + half_c, r1 + half_r),
               (c1 - half_c, r1 - half_r), (c0 - half_c, r0 - half_r)]
    return _from_pixels(corners, shape, pixel_size)


def _si_scale(largest, unit):
    """``(scale, label)`` for the SI prefix that suits ``largest``: 2e-5, "m" -> 1e-6, "µm"."""
    exponent = 0
    if np.isfinite(largest) and largest > 0:
        exponent = int(np.clip(3 * np.floor(np.log10(largest) / 3), -24, 24))
    return 10.0**exponent, f"{EngFormatter.ENG_PREFIXES[exponent]}{unit}"


def _scaled(scale):
    """Tick formatter showing ``value / scale`` as a plain number (rounded, no "-0")."""
    return FuncFormatter(lambda value, pos: f"{round(value / scale, 10) + 0.0:g}")


def _figure_and_axes(ax):
    """``(fig, ax)``: a new figure if ``ax`` is None, else ``ax`` and its figure."""
    import matplotlib.pyplot as plt

    if ax is None:
        return plt.subplots()
    return ax.get_figure(), ax
