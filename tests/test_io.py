"""AsylumScan on a committed Asylum AC-mode scan (512 x 512, four channels)."""

import numpy as np
import pytest
from igor2 import binarywave

from _paths import DATA
from pizprobe import AsylumScan
from pizprobe.io import _undo_saved_flatten

FILE = DATA / "dose_280_wg0002.ibw"
CHANNELS = ["HeightRetrace", "AmplitudeRetrace", "PhaseRetrace", "ZSensorRetrace"]


@pytest.fixture(scope="module")
def raw():
    return AsylumScan(FILE)


@pytest.fixture(scope="module")
def saved():
    return AsylumScan(FILE, undo_default_correction=False)


def test_channels_and_shapes(raw):
    assert raw.channels == CHANNELS
    for name in CHANNELS:
        assert raw[name].data.shape == (512, 512)
        assert name in raw
    assert raw["PhaseRetrace"].units == "deg"
    assert raw["ZSensorRetrace"].units == "m"


def test_unknown_channel_lists_available(raw):
    with pytest.raises(KeyError, match="ZSensorRetrace"):
        raw["Height"]


def test_saved_data_carries_asylum_flatten(saved):
    # An unmasked order-1 line flatten leaves every line with zero mean.
    for name in ("HeightRetrace", "ZSensorRetrace"):
        assert not saved[name].correction_undone
        np.testing.assert_allclose(saved[name].data.mean(axis=1), 0, atol=1e-12)


def test_undo_adds_back_stored_offsets_and_slopes(raw, saved):
    m, dx = raw.metadata, raw.pixel_size[0]
    x = np.arange(512) * dx
    for layer, name in ((0, "HeightRetrace"), (3, "ZSensorRetrace")):
        # Offsets/slopes are indexed in stored line order; the images are flipped.
        line = m[f"Flatten Offsets {layer}"][:, None] + m[f"Flatten Slopes {layer}"][:, None] * x
        np.testing.assert_allclose(raw[name].data - saved[name].data, np.flipud(line), atol=1e-15)
        assert raw[name].correction_undone
        assert raw[name].data.mean(axis=1).std() > 1e-9  # line offsets are back


def test_unflattened_channels_are_untouched(raw, saved):
    for name in ("AmplitudeRetrace", "PhaseRetrace"):
        assert not raw[name].correction_undone
        np.testing.assert_array_equal(raw[name].data, saved[name].data)


def test_orientation_matches_afmreader(saved):
    ibw = pytest.importorskip("AFMReader.ibw")
    image, _ = ibw.load_ibw(FILE, "HeightRetrace")  # nm, saved (flattened) data
    np.testing.assert_allclose(saved["HeightRetrace"].data * 1e9, image, rtol=1e-6, atol=1e-6)


def test_orientation_is_flipud_of_stored_transpose(saved):
    stored = binarywave.load(str(FILE))["wave"]["wData"][:, :, 2]
    np.testing.assert_array_equal(saved["PhaseRetrace"].data, np.flipud(stored.T))


def test_metadata_parsing(raw):
    m = raw.metadata
    assert m["ScanPoints"] == 512 and isinstance(m["ScanPoints"], int)
    assert m["ScanSize"] == pytest.approx(2e-5)
    assert m["ImagingMode"] == "AC Mode"
    assert m["Flatten Offsets 0"].shape == (512,)
    assert raw.size == pytest.approx((2e-5, 2e-5))
    assert raw.pixel_size == pytest.approx((2e-5 / 511,) * 2)


def test_summary_contents(raw):
    text = str(raw)
    for piece in ["20.00 × 20.00 µm", "512 × 512 px", "39.14 × 39.14 nm",
                  "0.511 Hz", "294.206 kHz", "undone on: HeightRetrace, ZSensorRetrace",
                  *CHANNELS]:
        assert piece in text


def test_unverified_corrections_raise():
    z = np.zeros((4, 4))
    with pytest.raises(NotImplementedError, match="undo_default_correction=False"):
        _undo_saved_flatten(z, {"FlattenOrder 0": 2}, 0, 1.0)
    with pytest.raises(NotImplementedError):
        _undo_saved_flatten(z, {"FlattenOrder 0": -1, "PlanefitOrder 0": 1}, 0, 1.0)


def test_flatten_order_zero_adds_offsets_only():
    z = np.zeros((2, 3))
    meta = {"FlattenOrder 0": 0, "Flatten Offsets 0": np.array([1.0, 2.0])}
    out, undone = _undo_saved_flatten(z, meta, 0, 1.0)
    assert undone
    np.testing.assert_array_equal(out, [[1, 1, 1], [2, 2, 2]])
