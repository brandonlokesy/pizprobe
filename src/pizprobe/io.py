# pizprobe/io.py
"""
Loaders for AFM scan files.

Classes
-------
Channel
    One image channel of a scan (Height, Phase, ...), with its units.
AsylumScan
    An Asylum Research ``.ibw`` scan: every channel, the parsed wave note, and a
    readable summary. Undoes the line flatten Asylum applies by default when saving.

Conventions
-----------
- Data are in SI units as stored in the file (lengths in m, phase in degrees).
- Images use the orientation of the Asylum display and of AFMReader's ``load_ibw``:
  ``np.flipud(wData.T)``, so rows are scan lines, the first one at the bottom when shown
  with ``imshow(origin="upper")``. Files saved with ``ScanDown: 1`` were checked only
  against AFMReader, not against the Asylum display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from igor2 import binarywave

# Channel name (without the Trace/Retrace suffix) -> units of the stored values.
_UNITS = {
    "Height": "m",
    "ZSensor": "m",
    "Amplitude": "m",
    "Deflection": "m",
    "Phase": "deg",
    "Frequency": "Hz",
}

_INT = re.compile(r"[+-]?\d+")


@dataclass(frozen=True, eq=False)
class Channel:
    """
    One image channel of a scan.

    Attributes
    ----------
    name : str
        Channel name as stored in the file, e.g. ``"ZSensorRetrace"``.
    data : np.ndarray
        2-D image, rows = scan lines, in SI units (see module docstring for orientation).
    units : str
        Units of ``data``: ``"m"``, ``"deg"``, ``"Hz"``, or ``""`` when unknown.
    correction_undone : bool
        True if the line flatten Asylum saved on this channel has been removed.
        False if the channel was never flattened, or if the undo was switched off.
    """

    name: str
    data: np.ndarray
    units: str
    correction_undone: bool

    def __repr__(self) -> str:
        undone = ", correction undone" if self.correction_undone else ""
        return f"<Channel {self.name!r} {self.data.shape} [{self.units}]{undone}>"


class AsylumScan:
    """
    An Asylum Research AFM scan loaded from an ``.ibw`` file.

    All channels are loaded; pick one explicitly with ``scan["ZSensorRetrace"]``.

    Parameters
    ----------
    path : str or Path
        The ``.ibw`` file.
    undo_default_correction : bool
        Asylum saves Height and ZSensor with a line flatten already applied (an offset
        and slope removed from every scan line, fitted without a mask). The removed
        offsets and slopes are stored in the wave note; if True (default), they are
        added back so the data are raw. Raises NotImplementedError for saved
        corrections whose convention has not been verified (flatten order >= 2, or a
        plane fit); pass False to load such files as saved.

    Attributes
    ----------
    path : Path
    metadata : dict
        Every ``key: value`` line of the wave note. Numbers become int/float,
        comma-separated lists become float arrays, everything else stays str.
    shape : tuple
        ``(n_rows, n_cols)`` of every channel.
    size : tuple
        ``(x, y)`` scan size in m (fast, slow axis).
    pixel_size : tuple
        ``(dx, dy)`` in m, from the wave scaling (``ScanSize / (points - 1)``).

    Examples
    --------
    >>> scan = AsylumScan("dose_280_wg0002.ibw")
    >>> print(scan)                         # summary of the scan settings
    >>> z = scan["ZSensorRetrace"].data     # metres
    """

    def __init__(self, path, undo_default_correction: bool = True):
        self.path = Path(path)
        wave = binarywave.load(str(self.path))["wave"]
        self.metadata = _parse_note(wave["note"])
        self.undo_default_correction = undo_default_correction

        stack = wave["wData"]
        if stack.ndim == 2:  # single-channel wave
            stack = stack[:, :, None]
            names = [wave["wave_header"]["bname"].decode()]
        else:
            names = [label.decode() for label in wave["labels"][2][1:]]

        dx, dy = (float(s) for s in wave["wave_header"]["sfA"][:2])
        self.pixel_size = (dx, dy)

        self._channels = {}
        for layer, name in enumerate(names):
            z = stack[:, :, layer].T.astype(float)  # rows = scan lines
            undone = False
            if undo_default_correction:
                z, undone = _undo_saved_flatten(z, self.metadata, layer, dx)
            self._channels[name] = Channel(
                name=name,
                data=np.flipud(z),
                units=_units(name),
                correction_undone=undone,
            )

        self.shape = next(iter(self._channels.values())).data.shape
        self.size = (
            float(self.metadata.get("FastScanSize", dx * (self.shape[1] - 1))),
            float(self.metadata.get("SlowScanSize", dy * (self.shape[0] - 1))),
        )

    @property
    def channels(self) -> list[str]:
        """Names of the channels in the file, in stored order."""
        return list(self._channels)

    def __getitem__(self, name: str) -> Channel:
        try:
            return self._channels[name]
        except KeyError:
            raise KeyError(
                f"No channel {name!r} in {self.path.name}. Available: {self.channels}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._channels

    def __repr__(self) -> str:
        return (
            f"<AsylumScan {self.path.name!r} {self.shape[0]}x{self.shape[1]}, "
            f"channels {self.channels}>"
        )

    def __str__(self) -> str:
        return self.summary()

    def summary(self) -> str:
        """Readable summary of the scan settings. Lines whose keys are missing are left out."""
        g = self.metadata.get
        lines = [f"AsylumScan  {self.path.name}"]

        def num(key):
            """Metadata value as a number, or None if missing or not numeric."""
            value = g(key)
            return value if isinstance(value, (int, float)) else None

        def add(label, text):
            lines.append(f"  {label:<12} {text}")

        acquired = []
        when = " ".join(str(s) for s in (g("Date"), g("Time")) if s)
        if when:
            acquired.append(when)
        if g("MicroscopeModel"):
            acquired.append(str(g("MicroscopeModel")))
        if g("Version"):
            acquired.append(f"AR {g('Version')}")
        if acquired:
            add("Acquired", " · ".join(acquired))

        mode = ", ".join(str(s) for s in (g("ImagingMode"), g("ScanMode")) if s)
        if mode:
            add("Mode", mode)

        sx, sy = self.size
        scan = [f"{sx * 1e6:.2f} × {sy * 1e6:.2f} µm", f"{self.shape[1]} × {self.shape[0]} px"]
        if num("ScanAngle") is not None:
            scan.append(f"angle {num('ScanAngle'):g}°")
        if num("ScanDown") is not None:
            scan.append("slow scan down" if num("ScanDown") else "slow scan up")
        add("Scan", " · ".join(scan))

        dx, dy = self.pixel_size
        add("Pixel size", f"{dx * 1e9:.2f} × {dy * 1e9:.2f} nm   "
                          "(sampling only; real lateral resolution is tip-limited)")

        timing = []
        if num("ScanRate") is not None:
            timing.append(f"{num('ScanRate'):.3g} Hz line rate")
        if g("Scan Time"):
            timing.append(f"{g('Scan Time')} total")
        if timing:
            add("Timing", " · ".join(timing))

        is_ac = "AC" in str(g("ImagingMode", ""))
        if is_ac:
            lever = []
            if num("DriveFrequency") is not None:
                text = f"drive {num('DriveFrequency') / 1e3:.3f} kHz"
                if num("ResFreq1"):
                    text += f" (tuned resonance {num('ResFreq1') / 1e3:.3f} kHz)"
                lever.append(text)
            if num("DriveAmplitude") is not None:
                lever.append(f"drive amplitude {num('DriveAmplitude'):.3f} V")
            if lever:
                add("Cantilever", " · ".join(lever))

        feedback = []
        if is_ac and num("AmplitudeSetpointVolts") is not None:
            text = f"setpoint {num('AmplitudeSetpointVolts'):.3f} V"
            free = num("FreeAirAmplitude")
            if free:
                text += f" / free air {free:.3f} V (ratio {num('AmplitudeSetpointVolts') / free:.2f})"
            feedback.append(text)
        elif not is_ac and num("DeflectionSetpointVolts") is not None:
            feedback.append(f"deflection setpoint {num('DeflectionSetpointVolts'):.3f} V")
        for key, label in (("IntegralGain", "I gain"), ("ProportionalGain", "P gain")):
            if num(key) is not None:
                feedback.append(f"{label} {num(key):g}")
        if feedback:
            add("Feedback", " · ".join(feedback))

        invols, k = num("InvOLS"), num("SpringConstant")
        if invols is not None and k is not None:
            text = f"InvOLS {invols * 1e9:.1f} nm/V · k {k:.2f} N/m"
            if np.isclose(invols, 1e-7) and np.isclose(k, 1.0):
                text += "   (!) software defaults: amplitudes in m are not calibrated"
            add("Calibration", text)

        bias = [f"{label} {num(key):g} V" for key, label in
                (("TipVoltage", "tip"), ("SurfaceVoltage", "surface")) if num(key) is not None]
        if bias:
            add("Bias", " · ".join(bias))

        add("Channels", " · ".join(
            f"{c.name} [{c.units or '?'}]" for c in self._channels.values()))

        undone = [c.name for c in self._channels.values() if c.correction_undone]
        if undone:
            add("Correction", "Asylum line flatten undone on: " + ", ".join(undone))
        else:
            saved = [n for i, n in enumerate(self.channels)
                     if int(g(f"FlattenOrder {i}", -1)) >= 0]
            add("Correction", ("Asylum line flatten still applied on: " + ", ".join(saved))
                if saved else "none saved by Asylum")
        return "\n".join(lines)


def _parse_note(note: bytes) -> dict:
    """Parse the ``key: value`` lines of an Asylum wave note. First occurrence of a key wins."""
    meta = {}
    for line in note.decode("latin-1").replace("\r\n", "\r").split("\r"):
        key, sep, value = line.partition(":")
        if sep and key.strip() and key.strip() not in meta:
            meta[key.strip()] = _parse_value(value.strip())
    return meta


def _parse_value(text: str):
    if "," in text:
        try:
            return np.array([float(v) for v in text.split(",") if v.strip()])
        except ValueError:
            return text
    if _INT.fullmatch(text):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def _units(name: str) -> str:
    base = re.sub(r"(Trace|Retrace)$", "", name)
    for prefix, units in _UNITS.items():
        if base.startswith(prefix):
            return units
    return ""


def _undo_saved_flatten(z: np.ndarray, meta: dict, layer: int, dx: float):
    """
    Add back the line flatten Asylum saved on data layer `layer`.

    `z` has rows = scan lines, in stored (not display) order. Asylum removes
    ``offset[row] + slope[row] * x`` from every line, with x = column * dx in m and the
    slope in m/m. Returns ``(z, undone)``.
    """
    order = int(meta.get(f"FlattenOrder {layer}", -1))
    planefit = int(meta.get(f"PlanefitOrder {layer}", -1))
    if planefit >= 0 or order >= 2:
        raise NotImplementedError(
            f"Layer {layer} was saved with flatten order {order} and plane fit order "
            f"{planefit}; undoing that is not implemented (only flatten order 0/1 is "
            "verified). Load with undo_default_correction=False."
        )
    if order < 0:
        return z, False

    offsets = np.atleast_1d(meta[f"Flatten Offsets {layer}"]).astype(float)
    if order >= 1:
        slopes = np.atleast_1d(meta[f"Flatten Slopes {layer}"]).astype(float)
    else:
        slopes = np.zeros_like(offsets)
    if len(offsets) != z.shape[0] or len(slopes) != z.shape[0]:
        raise ValueError(
            f"Layer {layer}: {len(offsets)} flatten offsets / {len(slopes)} slopes "
            f"for {z.shape[0]} scan lines."
        )
    x = np.arange(z.shape[1]) * dx
    return z + offsets[:, None] + slopes[:, None] * x, True
