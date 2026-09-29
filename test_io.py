import datetime

import numpy as np
import pytest

from asdspec import read_asd
from asdspec.io import DEFAULT_SPLICE1, DEFAULT_SPLICE2
from conftest import CHANNELS, write_asd


def test_roundtrip_header_and_spectrum(tmp_path, wl, solar):
    when = datetime.datetime(2024, 5, 24, 14, 3, 39)
    path = write_asd(tmp_path / "240524a.000", solar, when, it_ms=136)
    m = read_asd(path)

    assert m["datetime"] == when
    assert m["it"] == 136
    assert m["channels"] == CHANNELS
    assert m["type"] == "RAW"
    assert m["is_raw_dn"]
    assert m["index"] == 0
    np.testing.assert_allclose(m["wavelength"], wl)
    np.testing.assert_allclose(m["spectrum"], solar, rtol=1e-6)


def test_gain_and_offset_fields_are_not_zero(tmp_path, solar):
    """Guards the 128-byte GPS block. At 32 these fields read back as zero."""
    path = write_asd(tmp_path / "x.000", solar, datetime.datetime(2024, 1, 1, 12, 0),
                     it_ms=34, swir1_gain=4096, swir1_offset=2568)
    m = read_asd(path)
    assert m["it"] == 34
    assert m["swir1_gain"] == 4096
    assert m["swir1_offset"] == 2568


def test_unwritten_splice_fields_fall_back(tmp_path, solar):
    path = write_asd(tmp_path / "x.001", solar, datetime.datetime(2024, 1, 1, 12, 0))
    m = read_asd(path)
    assert m["splice1"] == DEFAULT_SPLICE1
    assert m["splice2"] == DEFAULT_SPLICE2


def test_rejects_non_asd(tmp_path):
    bad = tmp_path / "notes.000"
    bad.write_bytes(b"NOPE" + b"\x00" * 600)
    with pytest.raises(ValueError, match="not an ASD"):
        read_asd(bad)


def test_rejects_truncated(tmp_path, solar):
    path = write_asd(tmp_path / "x.002", solar, datetime.datetime(2024, 1, 1, 12, 0))
    path.write_bytes(path.read_bytes()[:900])
    with pytest.raises(ValueError, match="expected"):
        read_asd(path)
