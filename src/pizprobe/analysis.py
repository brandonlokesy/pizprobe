# pizprobe/analysis.py
"""
Measurements on a processed AFM image.

Classes
-------
Profile
    The result of one line cut: distance along the line and the values sampled on it.
Markers
    Two points on a line cut and the differences between them.
StepHeight
    Mean levels of two ranges on a line cut and the difference between them.
Crossings
    The distances where a line cut passes through a given value.

Functions
---------
profile
    Sample an image along a straight line (a line cut), optionally averaged over a band.
markers
    Values at two distances along a line cut, and their differences (like cursors).
step_height
    Mean value in two distance ranges along a line cut, and their difference.
crossings
    Distances where a line cut passes through a given value, e.g. to measure a width.

Conventions
-----------
- Line ends are ``(horizontal, vertical)`` points, with the same rule as
  ``pizprobe.mask.rectangle``: pixels (origin top left) without ``pixel_size``, metres
  (origin bottom left) with ``pixel_size=scan.pixel_size``.
- Several line cuts are a plain list of ``Profile``:
  ``[profile(z, s, e, pixel_size=px) for s, e in lines]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ._common import _as_image, _check_pixel_size, _to_pixels

# Pixel-coordinate tolerance for "end point inside the image", so an end given in metres
# exactly on the last pixel centre (up to rounding in y / dy) is accepted.
_EDGE = 1e-9


@dataclass(frozen=True, eq=False)
class Profile:
    """
    One line cut through an image.

    Attributes
    ----------
    distance : np.ndarray
        1-D, distance of each sample from ``start``: in m if ``pixel_size`` was given,
        else in pixels. Samples are evenly spaced, at most 1 px apart.
    values : np.ndarray
        1-D, image values at the samples (mean over the band if ``width > 1``), in the
        units of the image. NaN where no finite value was found.
    start, end : tuple of float
        Line ends as given, in pixels or m.
    pixel_size : tuple of float or None
        ``(dx, dy)`` in m if the ends were given in metres, else None.
    width : int
        Number of parallel lines averaged, 1 px apart.
    start_px, end_px : tuple of float
        Line ends in pixel coordinates ``(col, row)``, used to draw the line on an image.
    """

    distance: np.ndarray
    values: np.ndarray
    start: tuple
    end: tuple
    pixel_size: tuple | None
    width: int
    start_px: tuple
    end_px: tuple

    def __repr__(self) -> str:
        length = self.distance[-1]
        text = f"{length * 1e6:.3g} µm" if self.pixel_size is not None else f"{length:.1f} px"
        return f"<Profile {len(self.values)} points, {text}, width {self.width}>"


def profile(z, start, end, pixel_size=None, width: int = 1) -> Profile:
    """
    Sample an image along a straight line (a line cut).

    Samples are evenly spaced from ``start`` to ``end``, at most 1 px apart, and each
    value is interpolated bilinearly from the 4 nearest pixels. With ``width > 1``, the
    values are the mean of ``width`` parallel lines 1 px apart, centred on the line;
    this lowers the noise by about ``sqrt(width)`` but also averages any change of the
    surface across the band.

    Parameters
    ----------
    z : np.ndarray
        2-D image, e.g. a levelled height image.
    start, end : tuple of float
        ``(horizontal, vertical)`` ends of the line, both inside the image. Pixels
        (origin top left) without ``pixel_size``; metres (origin bottom left) with it.
    pixel_size : tuple of float, optional
        ``(dx, dy)`` in m, e.g. ``scan.pixel_size``. Also sets the unit of
        ``Profile.distance`` (m instead of px).
    width : int
        Number of parallel lines averaged, >= 1. The band is perpendicular to the line
        in pixel coordinates (the same as on the sample for square pixels). Points of
        the band outside the image are left out of the mean.

    Returns
    -------
    Profile

    Raises
    ------
    ValueError
        If ``z`` is not 2-D, an end is outside the image, ``start == end``, or
        ``width`` is not an integer >= 1.

    Examples
    --------
    >>> p = profile(flat, (2e-6, 10e-6), (18e-6, 10e-6), pixel_size=scan.pixel_size, width=5)
    >>> p.distance, p.values          # m, m
    """
    z = _as_image(z)
    if int(width) != width or width < 1:
        raise ValueError(f"width must be an integer >= 1, got {width!r}.")
    width = int(width)
    if pixel_size is not None:
        pixel_size = _check_pixel_size(pixel_size)

    (c0, r0), (c1, r1) = _to_pixels([start, end], z.shape, pixel_size)
    n_rows, n_cols = z.shape
    for name, given, c, r in (("start", start, c0, r0), ("end", end, c1, r1)):
        if not (-_EDGE <= c <= n_cols - 1 + _EDGE and -_EDGE <= r <= n_rows - 1 + _EDGE):
            raise ValueError(
                f"{name} {tuple(given)} is outside the image (pixel position col {c:.2f}, "
                f"row {r:.2f}; image is {n_cols} x {n_rows} px)."
            )

    dc, dr = c1 - c0, r1 - r0
    length_px = np.hypot(dc, dr)
    if length_px == 0:
        raise ValueError("start and end are the same point.")

    t = np.linspace(0.0, 1.0, int(np.ceil(length_px)) + 1)
    offsets = np.arange(width) - (width - 1) / 2
    normal_c, normal_r = -dr / length_px, dc / length_px
    cols = c0 + t[None, :] * dc + offsets[:, None] * normal_c
    rows = r0 + t[None, :] * dr + offsets[:, None] * normal_r
    band = ndimage.map_coordinates(
        z, [rows.ravel(), cols.ravel()], order=1, mode="constant", cval=np.nan
    ).reshape(width, len(t))

    finite = np.isfinite(band)
    count = finite.sum(axis=0)
    total = np.where(finite, band, 0.0).sum(axis=0)
    values = np.full(len(t), np.nan)
    values[count > 0] = total[count > 0] / count[count > 0]

    if pixel_size is None:
        length = length_px
    else:
        dx, dy = pixel_size
        length = np.hypot(dc * dx, dr * dy)

    return Profile(
        distance=t * length,
        values=values,
        start=tuple(float(v) for v in start),
        end=tuple(float(v) for v in end),
        pixel_size=pixel_size,
        width=width,
        start_px=(float(c0), float(r0)),
        end_px=(float(c1), float(r1)),
    )


@dataclass(frozen=True, eq=False)
class Markers:
    """
    Two points on a line cut, like a pair of cursors.

    Attributes
    ----------
    distance : np.ndarray
        ``(d1, d2)`` along the profile, in the units of ``Profile.distance`` (m or px).
    values : np.ndarray
        ``(z1, z2)``, the profile values at those distances, in the units of the image.
    profile : Profile
        The line cut the markers were placed on, so ``plotting.profiles`` can draw them on
        it (``measurements=``).
    """

    distance: np.ndarray
    values: np.ndarray
    profile: Profile

    @property
    def dx(self) -> float:
        """Lateral distance ``d2 - d1``."""
        return float(self.distance[1] - self.distance[0])

    @property
    def dz(self) -> float:
        """Value difference ``z2 - z1`` (a height difference on a height image)."""
        return float(self.values[1] - self.values[0])

    def __repr__(self) -> str:
        return f"<Markers dx {self.dx:.4g}, dz {self.dz:.4g}>"


def markers(p: Profile, distances) -> Markers:
    """
    Values at two distances along a line cut, and their differences.

    Each value is interpolated linearly between the two nearest samples of the profile.
    A single point carries the full noise of the profile; for the height of a flat level,
    an average over a range is less noisy.

    Parameters
    ----------
    p : Profile
        A line cut from ``profile``.
    distances : pair of float
        ``(d1, d2)`` along the line, measured from ``p.start``, in the units of
        ``p.distance``: m if the profile was made with ``pixel_size``, else px. Read them
        off the profile plot (a tick "2.5" on "distance (µm)" is ``2.5e-6``).

    Returns
    -------
    Markers
        ``.distance``, ``.values``, and the differences ``.dx`` (d2 - d1) and ``.dz``
        (z2 - z1). Draw them with ``plotting.profiles(..., measurements=[m])``.

    Raises
    ------
    ValueError
        If ``distances`` is not two numbers, a distance is outside the profile, or the
        profile has no finite value there.

    Examples
    --------
    >>> m = markers(cut, (7.5e-6, 8.5e-6))
    >>> m.dx, m.dz          # m, m
    """
    d = _check_distances(p, distances, "distances", "(d1, d2)")
    values = np.interp(d, p.distance, p.values)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"The profile has no finite value at distance {d[~np.isfinite(values)][0]:g} "
            "(the line cut passes over NaN points there)."
        )
    return Markers(distance=d, values=values, profile=p)


@dataclass(frozen=True, eq=False)
class StepHeight:
    """
    Mean levels of two ranges on a line cut, e.g. the top and the floor of a step.

    Attributes
    ----------
    ranges : np.ndarray
        ``[[a0, a1], [b0, b1]]``, the two distance ranges, each sorted, in the units of
        ``Profile.distance`` (m or px).
    levels : np.ndarray
        ``(level_a, level_b)``, the mean of the finite profile values in each range, in the
        units of the image.
    std : np.ndarray
        Standard deviation of the values in each range: the roughness or noise of each
        level, not the uncertainty of its mean (for that, divide by ``sqrt(n)``, which is
        only right if the samples are independent).
    n : np.ndarray
        Number of profile samples used in each range.
    profile : Profile
        The line cut the ranges were placed on, so ``plotting.profiles`` can draw them on
        it (``measurements=``).
    """

    ranges: np.ndarray
    levels: np.ndarray
    std: np.ndarray
    n: np.ndarray
    profile: Profile

    @property
    def dz(self) -> float:
        """Level difference ``level_b - level_a``."""
        return float(self.levels[1] - self.levels[0])

    def __repr__(self) -> str:
        return f"<StepHeight dz {self.dz:.4g}, n {self.n[0]} and {self.n[1]}>"


def step_height(p: Profile, a, b) -> StepHeight:
    """
    Mean value in two distance ranges along a line cut, and their difference.

    Each level is the plain mean of the profile samples whose distance lies inside the
    range (NaN samples left out), so each range should cover one flat level only: stop
    it short of the sidewall, where the tip shape and the sidewall slope mix the two
    levels. The levels are not fitted with a slope, so any tilt left along the cut
    changes the result by the tilt times the distance between the ranges; level the
    image first.

    Parameters
    ----------
    p : Profile
        A line cut from ``profile``.
    a, b : pair of float
        ``(start, end)`` of each range along the line, in either order, measured from
        ``p.start`` in the units of ``p.distance``: m if the profile was made with
        ``pixel_size``, else px. Both ends are included.

    Returns
    -------
    StepHeight
        ``.levels``, ``.std``, ``.n`` for ranges a and b, and ``.dz`` (level b - level a).
        Draw it with ``plotting.profiles(..., measurements=[s])``.

    Raises
    ------
    ValueError
        If a range is not two numbers, reaches outside the profile, or holds no finite
        profile sample.

    Examples
    --------
    >>> s = step_height(cut, a=(1e-6, 6e-6), b=(12e-6, 18e-6))   # top, then floor
    >>> s.dz                   # m, negative: b is below a
    >>> s.levels, s.std, s.n
    """
    ranges = np.sort([_check_distances(p, a, "a", "(start, end)"),
                      _check_distances(p, b, "b", "(start, end)")], axis=1)
    levels, std, n = [], [], []
    for name, (lo, hi) in zip("ab", ranges):
        inside = (p.distance >= lo) & (p.distance <= hi) & np.isfinite(p.values)
        if not inside.any():
            raise ValueError(
                f"Range {name} ({lo:g} to {hi:g}) holds no finite profile sample: samples "
                f"are {p.distance[1]:g} apart, so make the range wider or move it off NaN "
                "points."
            )
        levels.append(p.values[inside].mean())
        std.append(p.values[inside].std())
        n.append(int(inside.sum()))
    return StepHeight(ranges=ranges, levels=np.array(levels), std=np.array(std),
                      n=np.array(n), profile=p)


@dataclass(frozen=True, eq=False)
class Crossings:
    """
    The distances where a line cut passes through a given value.

    Attributes
    ----------
    distance : np.ndarray
        1-D, sorted, the distance of each crossing from ``Profile.start``, in the units of
        ``Profile.distance`` (m or px). Empty if the profile never reaches ``level``.
    level : float
        The value crossed, in the units of the image.
    profile : Profile
        The line cut, so ``plotting.profiles`` can draw the crossings on it
        (``measurements=``).
    """

    distance: np.ndarray
    level: float
    profile: Profile

    def __repr__(self) -> str:
        return f"<Crossings {len(self.distance)} at level {self.level:.4g}>"


def crossings(p: Profile, level: float) -> Crossings:
    """
    Distances where a line cut passes through ``level``.

    Between two neighbouring samples on opposite sides of ``level``, the crossing is
    placed by straight-line interpolation. The width of a feature at a chosen height is
    the distance between two crossings. On a sloped sidewall (fabrication, or the tip
    shape) the width depends on ``level``: a ridge measures wider near its foot and
    narrower near its top. Choose the level yourself, e.g. half-way between the two
    levels of ``step_height``.

    Parameters
    ----------
    p : Profile
        A line cut from ``profile``.
    level : float
        The value to cross, in the units of the image (m for heights).

    Returns
    -------
    Crossings
        ``.distance`` of every crossing, in order along the line. Noise or particles near
        ``level`` give extra crossings: check them on the plot with
        ``plotting.profiles(..., measurements=[x])``. A sample exactly at ``level``
        counts as above it. A crossing hidden in a stretch of NaN samples is not found.

    Raises
    ------
    ValueError
        If ``level`` is not a finite number.

    Examples
    --------
    >>> s = step_height(cut, a=(1e-6, 6e-6), b=(12e-6, 18e-6))
    >>> x = crossings(cut, level=s.levels[1] + 0.5 * (s.levels[0] - s.levels[1]))
    >>> x.distance                              # m
    >>> width = x.distance[1] - x.distance[0]   # e.g. between the two sidewalls of a ridge
    """
    level = float(level)
    if not np.isfinite(level):
        raise ValueError(f"level must be a finite number, got {level!r}.")
    v = p.values - level
    above = v >= 0
    # Neighbouring samples, both finite, on opposite sides of the level.
    pair = np.isfinite(v[:-1]) & np.isfinite(v[1:]) & (above[:-1] != above[1:])
    i = np.nonzero(pair)[0]
    fraction = v[i] / (v[i] - v[i + 1])             # 0 at sample i, 1 at sample i + 1
    distance = p.distance[i] + fraction * (p.distance[i + 1] - p.distance[i])
    return Crossings(distance=distance, level=level, profile=p)


def _check_distances(p: Profile, pair, name: str, form: str) -> np.ndarray:
    """``pair`` as 2 floats, each inside ``[0, p.distance[-1]]``."""
    d = np.asarray(pair, dtype=float)
    if d.shape != (2,):
        raise ValueError(f"{name} must be two numbers {form}, got {pair!r}.")
    length = p.distance[-1]
    for value in d:
        if not -_EDGE * length <= value <= length * (1 + _EDGE):
            raise ValueError(
                f"{name}: distance {value:g} is outside the profile, which runs from 0 to "
                f"{length:g} (in the units of Profile.distance)."
            )
    return d
