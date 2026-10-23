# pizprobe/loaders.py
"""
Loaders for AFM scan files.

Classes
-------
Channel
    One image channel of a scan (Height, Phase, ...), with its units.
AsylumScan
    An Asylum Research ``.ibw`` scan: every channel, the parsed wave note, and a
    readable summary. Undoes the line flatten Asylum applies by default when saving.
BrukerScan
    A Bruker NanoScope ``.spm`` scan: every channel, the parsed header, and a readable
    summary. Data are loaded as saved.

Conventions
-----------
- Data are in SI units (lengths in m, forces in N, phase in degrees).
- Images use the orientation of the Asylum display and of AFMReader's ``load_ibw``:
  ``np.flipud(wData.T)``, so rows are scan lines, the first one at the bottom when shown
  with ``imshow(origin="upper")``. Files saved with ``ScanDown: 1`` were checked only
  against AFMReader, not against the Asylum display.
- NanoScope images are ``np.flipud`` of the stored line order, as in Gwyddion's reader,
  whatever the ``Frame direction``. Not yet checked against the NanoScope display.
- Pixel size follows each manufacturer: Asylum stores ``ScanSize / (points - 1)`` (a
  pixel is a sample point, the first at 0 and the last at ScanSize); Bruker uses
  ``ScanSize / points`` (a pixel is a cell of that width).
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

# NanoScope header: ``\key: value``. The key may itself hold a colon (``@2:Z scale``),
# so the key ends at the first colon followed by whitespace or the end of the line.
_BRUKER_LINE = re.compile(r"(.+?):(?:\s+|$)(.*)")
# ``@2:Z scale`` value, e.g. ``V [Sens. ZsensSens] (0.00000000572205 V/LSB) 24.57600 V``.
_BRUKER_Z_SCALE = re.compile(r"\[Sens\. (?P<ref>[^\]]+)\] \((?P<lsb>\S+) V/LSB\)")
# ``@Sens.`` value, e.g. ``V 199.0681 nm/V``.
_BRUKER_SENS = re.compile(r"V\s+(?P<value>\S+)\s*(?P<unit>\S*)")
# Unit prefixes used in NanoScope headers; ``~`` stands for µ.
_PREFIX = {"": 1.0, "k": 1e3, "m": 1e-3, "~": 1e-6, "u": 1e-6, "µ": 1e-6, "n": 1e-9,
           "p": 1e-12, "f": 1e-15}
_SI_UNITS = ("m", "N", "V", "A")


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


class BrukerScan:
    """
    A Bruker AFM scan loaded from a NanoScope ``.spm`` file (NanoScope 9.x image files).

    All channels are loaded; pick one explicitly with ``scan["Height Sensor Retrace"]``.
    Channel names are the NanoScope name followed by the line direction.

    Data are loaded as saved. NanoScope records no line flatten in the header of a
    captured image, so none is undone. A file flattened after capture (in NanoScope
    Analysis) and saved again is not marked in its header either; ``summary()`` points
    out channels whose every line has zero mean, which is the trace such a flatten leaves.

    Parameters
    ----------
    path : str or Path
        The ``.spm`` file.

    Attributes
    ----------
    path : Path
    metadata : dict
        Every ``key: value`` line of the header outside the image sections. Numbers
        become int/float, everything else stays str.
    channel_metadata : dict
        ``{channel name: dict}``, the header section of each channel, parsed the same way.
    shape : tuple
        ``(n_rows, n_cols)`` of every channel. An interrupted scan has fewer rows than
        the planned ``metadata["Lines"]``.
    size : tuple
        ``(x, y)`` size of the scanned area in m (fast, slow axis).
    pixel_size : tuple
        ``(dx, dy)`` in m: scan size / planned number of points along each axis.

    Examples
    --------
    >>> scan = BrukerScan("sample6_image1.0_00000.spm")
    >>> print(scan)                                 # summary of the scan settings
    >>> z = scan["Height Sensor Retrace"].data      # metres
    """

    def __init__(self, path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.metadata, sections = _parse_bruker_header(raw, self.path.name)

        self._channels = {}
        self.channel_metadata = {}
        for section in sections:
            name = _bruker_channel_name(section)
            if name in self._channels:
                raise ValueError(f"{self.path.name}: two channels named {name!r}.")
            factor, units = _bruker_scale(section, self.metadata, name)
            z = _bruker_read_image(raw, section, name) * factor
            self.channel_metadata[name] = section
            self._channels[name] = Channel(
                name=name, data=np.flipud(z), units=units, correction_undone=False
            )

        first = sections[0]
        self.shape = next(iter(self._channels.values())).data.shape
        x, y = _bruker_length(first["Scan Size"], name="Scan Size")
        planned_lines = int(self.metadata.get("Lines", self.shape[0]))
        self.pixel_size = (x / int(first["Samps/line"]), y / planned_lines)
        self.size = (self.pixel_size[0] * self.shape[1], self.pixel_size[1] * self.shape[0])

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
            f"<BrukerScan {self.path.name!r} {self.shape[0]}x{self.shape[1]}, "
            f"channels {self.channels}>"
        )

    def __str__(self) -> str:
        return self.summary()

    def summary(self) -> str:
        """Readable summary of the scan settings. Lines whose keys are missing are left out."""
        g = self.metadata.get
        first = next(iter(self.channel_metadata.values()))
        lines = [f"BrukerScan  {self.path.name}"]

        def add(label, text):
            lines.append(f"  {label:<12} {text}")

        acquired = [str(s) for s in (g("Date"), g("Description")) if s]
        if g("Version"):
            acquired.append(f"NanoScope {g('Version')}")
        if acquired:
            add("Acquired", " · ".join(acquired))

        mode = [_quoted(g(key)) for key in ("@MicroscopeList", "@2:SPMFeedbackList")]
        mode = [m for m in mode if m]
        if mode:
            add("Mode", ", ".join(mode) + (" feedback" if len(mode) == 2 else ""))

        sx, sy = self.size
        scan = [f"{sx * 1e6:.2f} × {sy * 1e6:.2f} µm", f"{self.shape[1]} × {self.shape[0]} px"]
        if isinstance(g("Rotate Ang."), (int, float)):
            scan.append(f"angle {g('Rotate Ang.'):g}°")
        if first.get("Frame direction"):
            scan.append(f"frame {str(first['Frame direction']).lower()}")
        planned = g("Lines")
        if isinstance(planned, int) and self.shape[0] < planned:
            scan.append(f"interrupted: {self.shape[0]} of {planned} lines")
        add("Scan", " · ".join(scan))

        dx, dy = self.pixel_size
        add("Pixel size", f"{dx * 1e9:.2f} × {dy * 1e9:.2f} nm   "
                          "(sampling only; real lateral resolution is tip-limited)")

        if isinstance(g("Scan Rate"), (int, float)):
            add("Timing", f"{g('Scan Rate'):.3g} Hz line rate")

        calibration = []
        for ref, label in (("DeflSens", "deflection"), ("ForceDeflSens", "force")):
            sens = _BRUKER_SENS.match(str(g(f"@Sens. {ref}", "")))
            if sens:
                calibration.append(f"{label} {float(sens['value']):.2f} {sens['unit']}")
        if calibration:
            add("Calibration", " · ".join(calibration))

        add("Channels", " · ".join(
            f"{c.name} [{c.units or '?'}]" for c in self._channels.values()))

        flattened = [c.name for c in self._channels.values() if _lines_zero_mean(c.data)]
        if flattened:
            add("Correction", "every line has zero mean on: " + ", ".join(flattened)
                + " (looks line-flattened after capture; cannot be undone)")
        else:
            add("Correction", "none recorded by NanoScope")
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


def _parse_bruker_header(raw: bytes, filename: str):
    """
    Parse a NanoScope header. Returns ``(metadata, sections)``: the file-level
    ``key: value`` pairs, and one dict per ``\\*Ciao image list`` section (one per channel).
    First occurrence of a key wins within each dict.
    """
    end = raw.find(b"\x1a")
    if not raw.startswith(b"\\*File list") or end < 0:
        raise ValueError(
            f"{filename} is not a NanoScope image file: it should start with a "
            r"'\*File list' text header ending in byte 0x1A."
        )
    meta, sections = {}, []
    current = meta
    for line in raw[:end].decode("latin-1").split("\r\n"):
        line = line.lstrip("\\")
        if line.startswith("*"):  # section title
            if line == "*File list end":
                break
            if line == "*Ciao image list":
                current = {}
                sections.append(current)
            else:
                current = meta
            continue
        match = _BRUKER_LINE.fullmatch(line)
        if match and match[1].strip() not in current:
            current[match[1].strip()] = _parse_value(match[2].strip())
    if not sections:
        raise ValueError(f"{filename} holds no image channel (no '*Ciao image list').")
    return meta, sections


def _bruker_channel_name(section: dict) -> str:
    """NanoScope channel name followed by the line direction, e.g. "Height Sensor Retrace"."""
    name = _quoted(section.get("@2:Image Data")) or "Unnamed"
    direction = section.get("Line Direction")
    return f"{name} {direction}" if direction else name


def _quoted(text) -> str:
    """The ``"..."`` part of a NanoScope value such as ``S [ZSensor] "Height Sensor"``."""
    match = re.search(r'"(.*)"', str(text or ""))
    return match[1] if match else ""


def _bruker_scale(section: dict, meta: dict, name: str):
    """
    Factor from stored integers to SI units, and those units, for one channel.

    value = integer × (V/LSB written in ``@2:Z scale``) × ``@Sens.`` sensitivity. The V/LSB
    is taken as written, never recomputed from ``Bytes/pixel``, which NanoScope 9.x
    states as 2 for channels stored on 4 bytes.
    """
    z_scale = str(section.get("@2:Z scale", ""))
    match = _BRUKER_Z_SCALE.search(z_scale)
    if not match:
        raise NotImplementedError(
            f"Channel {name!r}: Z scale {z_scale!r} does not refer to a sensitivity "
            "('[Sens. ...]'); converting it is not implemented."
        )
    sens_text = meta.get(f"@Sens. {match['ref']}")
    sens = _BRUKER_SENS.match(str(sens_text or ""))
    if not sens:
        raise ValueError(
            f"Channel {name!r}: sensitivity 'Sens. {match['ref']}' is missing from the header."
        )
    factor = float(match["lsb"]) * float(sens["value"])

    numerator, _, denominator = sens["unit"].partition("/")
    for unit in _SI_UNITS:
        prefix = numerator[: -len(unit)] if numerator.endswith(unit) else None
        if denominator == "V" and prefix in _PREFIX:
            return factor * _PREFIX[prefix], unit
    return factor, ""  # e.g. "1/Arb" or no unit: kept as volts × sensitivity


def _bruker_read_image(raw: bytes, section: dict, name: str) -> np.ndarray:
    """
    The stored integers of one channel, as floats, rows = scan lines in stored order.

    The integer width is ``Data length / pixels``, not ``Bytes/pixel`` (see
    ``_bruker_scale``). Raises if a plane fit was saved with the channel.
    """
    plane = [float(v) for v in str(section.get("Plane fit", "0 0 0")).split()[:3]]
    if any(plane):
        raise NotImplementedError(
            f"Channel {name!r} was saved with plane fit coefficients {plane}; "
            "undoing them is not implemented."
        )
    n_cols, n_rows = int(section["Samps/line"]), int(section["Number of lines"])
    offset, length = int(section["Data offset"]), int(section["Data length"])
    width, rest = divmod(length, n_cols * n_rows)
    if rest or width not in (2, 4):
        raise ValueError(
            f"Channel {name!r}: {length} bytes for {n_rows} x {n_cols} pixels is not "
            "2 or 4 bytes per pixel."
        )
    if offset + length > len(raw):
        raise ValueError(
            f"Channel {name!r}: data run to byte {offset + length}, past the end of the "
            f"file ({len(raw)} bytes). The file may be truncated."
        )
    stored = np.frombuffer(raw, dtype=f"<i{width}", count=n_rows * n_cols, offset=offset)
    return stored.reshape(n_rows, n_cols).astype(float)


def _bruker_length(text: str, name: str):
    """``"10 10 ~m"`` -> ``(1e-5, 1e-5)`` in m."""
    *values, unit = str(text).split()
    prefix = unit[:-1] if unit.endswith("m") else None
    if len(values) != 2 or prefix not in _PREFIX:
        raise ValueError(f"Cannot read {name} {text!r} as two lengths with a unit.")
    return tuple(float(v) * _PREFIX[prefix] for v in values)


def _lines_zero_mean(z: np.ndarray) -> bool:
    """True if every line of ``z`` has zero mean, to 1e-6 of the image's range."""
    span = np.ptp(z)
    return bool(span > 0 and np.all(np.abs(z.mean(axis=1)) <= 1e-6 * span))
