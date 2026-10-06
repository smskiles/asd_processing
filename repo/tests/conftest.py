import datetime
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asdspec.io import ASD_FIELDS, ASD_STRUCT, HEADER_BYTES  # noqa: E402

CHANNELS = 2151


def write_asd(path, spectrum, when, it_ms=68, data_type=0, swir1_gain=4096,
              swir1_offset=1544, swir2_gain=2816, swir2_offset=1024):
    """Write a minimal but structurally valid ASD binary file."""
    values = dict.fromkeys(ASD_FIELDS, 0)
    values["co"] = b"ASD"
    values["comments"] = b""
    tm = (when.second, when.minute, when.hour, when.day, when.month - 1,
          when.year - 1900, 0, 0, 0)
    values["when"] = struct.pack("<9h", *tm)
    values["program_version"] = 100
    values["file_version"] = 68
    values["dc_corr"] = 1
    values["data_type"] = data_type
    values["ch1_wavel"] = 350.0
    values["wavel_step"] = 1.0
    values["data_format"] = 0
    values["channels"] = CHANNELS
    values["app_data"] = b""
    values["gps_data"] = b""
    values["it"] = it_ms
    values["swir1_gain"] = swir1_gain
    values["swir2_gain"] = swir2_gain
    values["swir1_offset"] = swir1_offset
    values["swir2_offset"] = swir2_offset
    values["smart_units"] = b""
    values["flags"] = b""
    values["spare"] = b""

    header = struct.pack(ASD_STRUCT, *[values[f] for f in ASD_FIELDS])
    assert len(header) == HEADER_BYTES
    path.write_bytes(header + np.asarray(spectrum, dtype="<f4").tobytes())
    return path


@pytest.fixture
def wl():
    return np.arange(350, 350 + CHANNELS, dtype=float)


@pytest.fixture
def solar(wl):
    """A smooth, solar-like spectrum: bright in the visible, falling into the SWIR."""
    return 1000 * np.exp(-((wl - 500) / 900) ** 2) + 80 * np.exp(-((wl - 1600) / 400) ** 2)


@pytest.fixture
def snowy(wl):
    """A smooth, snow-like albedo: high in the visible, near zero past 1400 nm."""
    return np.clip(0.9 / (1 + np.exp((wl - 1150) / 130)), 0.01, None)
