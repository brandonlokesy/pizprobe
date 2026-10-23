"""BrukerScan on a committed NanoScope 9.40 PeakForce scan (256 x 256, two channels)."""

import numpy as np
import pytest

from _paths import DATA
from pizprobe import BrukerScan
from pizprobe.loaders import _lines_zero_mean

FILE = DATA / "sample6_image1.0_00000.spm"
CHANNELS = ["Height Sensor Retrace", "Peak Force Error Retrace"]


@pytest.fixture(scope="module")
def scan():
    return BrukerScan(FILE)


def _write_spm(path, data, *, n_lines_planned=None, plane_fit="0 0 0 5",
               sens_unit="nm/V", bytes_per_pixel_stated=2):
    """A minimal NanoScope file holding one channel of int16 or int32 ``data``."""
    n_rows, n_cols = data.shape
    width = data.dtype.itemsize
    header_length = 4096
    header = "\r\n".join([
        r"\*File list",
        r"\Version: 0x09400202",
        r"\*Ciao scan list",
        r"\@Sens. ZsensSens: V 100.0000 " + sens_unit,
        rf"\Lines: {n_lines_planned or n_rows}",
        r"\*Ciao image list",
        rf"\Data offset: {header_length}",
        rf"\Data length: {data.size * width}",
        rf"\Bytes/pixel: {bytes_per_pixel_stated}",
        rf"\Plane fit: {plane_fit}",
        rf"\Samps/line: {n_cols}",
        rf"\Number of lines: {n_rows}",
        r"\Scan Size: 1 1 ~m",
        r"\Line Direction: Trace",
        r'\@2:Image Data: S [ZSensor] "Height Sensor"',
        r"\@2:Z scale: V [Sens. ZsensSens] (0.001 V/LSB) 24.57600 V",
        r"\*File list end",
        "\x1a",
    ]).encode("latin-1")
    path.write_bytes(header.ljust(header_length, b"\0") + data.astype(f"<i{width}").tobytes())
    return path


def test_channels_shapes_and_units(scan):
    assert scan.channels == CHANNELS
    for name in CHANNELS:
        assert scan[name].data.shape == (256, 256)
        assert name in scan
        assert not scan[name].correction_undone
    assert scan["Height Sensor Retrace"].units == "m"
    assert scan["Peak Force Error Retrace"].units == "N"


def test_unknown_channel_lists_available(scan):
    with pytest.raises(KeyError, match="Height Sensor Retrace"):
        scan["Height"]


def test_values_in_si(scan):
    # integer × V/LSB × sensitivity: 5.722e-9 V × 199.07 nm/V for height,
    # 3.75e-4 V × 39.2 nN/V for Peak Force Error.
    assert np.ptp(scan["Height Sensor Retrace"].data) == pytest.approx(27.227e-9, rel=1e-4)
    # Stated as 2 bytes/pixel but stored on 4: the written V/LSB must be used as is.
    assert np.ptp(scan["Peak Force Error Retrace"].data) == pytest.approx(3.8514e-9, rel=1e-4)


def test_orientation_is_flipud_of_stored_lines(scan):
    section = scan.channel_metadata["Height Sensor Retrace"]
    stored = np.frombuffer(FILE.read_bytes(), "<i4", 256 * 256, section["Data offset"])
    stored = stored.reshape(256, 256)
    ratio = scan["Height Sensor Retrace"].data / np.flipud(stored)
    np.testing.assert_allclose(ratio, ratio[0, 0])


def test_size_and_pixel_size(scan):
    assert scan.size == pytest.approx((2e-6, 2e-6))
    assert scan.pixel_size == pytest.approx((2e-6 / 256,) * 2)


def test_metadata_parsing(scan):
    m = scan.metadata
    assert m["Lines"] == 256 and isinstance(m["Lines"], int)
    assert m["Scan Rate"] == pytest.approx(1.46974)
    assert m["@Sens. ZsensSens"] == "V 199.0681 nm/V"
    section = scan.channel_metadata["Peak Force Error Retrace"]
    assert section["Bytes/pixel"] == 2
    assert section["@2:Image Data"] == 'S [] "Peak Force Error"'


def test_summary_contents(scan):
    text = str(scan)
    for piece in ["2.00 × 2.00 µm", "256 × 256 px", "7.81 × 7.81 nm", "1.47 Hz",
                  "ScanAsyst, Peak Force feedback", "none recorded by NanoScope", *CHANNELS]:
        assert piece in text


def test_interrupted_scan_keeps_planned_pixel_size(tmp_path):
    data = np.arange(4 * 8, dtype=np.int16).reshape(4, 8)
    s = BrukerScan(_write_spm(tmp_path / "cut.spm", data, n_lines_planned=8))
    assert s.shape == (4, 8)
    assert s.pixel_size == pytest.approx((1e-6 / 8,) * 2)
    assert s.size == pytest.approx((1e-6, 0.5e-6))
    assert "interrupted: 4 of 8 lines" in s.summary()
    np.testing.assert_allclose(s["Height Sensor Trace"].data, np.flipud(data) * 0.1e-9)


def test_unknown_sensitivity_unit_kept_as_volts(tmp_path):
    data = np.ones((2, 2), dtype=np.int32)
    s = BrukerScan(_write_spm(tmp_path / "arb.spm", data, sens_unit="1/Arb"))
    assert s["Height Sensor Trace"].units == ""
    np.testing.assert_allclose(s["Height Sensor Trace"].data, 0.1)


def test_saved_plane_fit_raises(tmp_path):
    path = _write_spm(tmp_path / "pf.spm", np.zeros((2, 2), np.int16), plane_fit="0.1 0 0 5")
    with pytest.raises(NotImplementedError, match="plane fit"):
        BrukerScan(path)


def test_not_a_nanoscope_file_raises(tmp_path):
    path = tmp_path / "x.spm"
    path.write_bytes(b"not a header")
    with pytest.raises(ValueError, match="not a NanoScope image file"):
        BrukerScan(path)


def test_truncated_file_raises(tmp_path):
    path = _write_spm(tmp_path / "t.spm", np.zeros((4, 4), np.int32))
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="truncated"):
        BrukerScan(path)


def test_lines_zero_mean():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(8, 16)) + np.arange(8)[:, None]
    assert not _lines_zero_mean(z)
    assert _lines_zero_mean(z - z.mean(axis=1, keepdims=True))
    assert not _lines_zero_mean(np.zeros((4, 4)))
