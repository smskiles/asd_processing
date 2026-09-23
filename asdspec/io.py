"""Reading ASD FieldSpec binary spectra."""

from __future__ import annotations

import datetime
import struct
from pathlib import Path

import numpy as np

# ASD binary layout. The file is a fixed 484-byte header followed by `channels`
# values. Note the GPS block is 128 bytes: several published parsers use 32,
# which shifts the integration time and SWIR gain/offset fields 96 bytes early so
# they silently read as zero. Those fields are needed for a correct albedo.
ASD_STRUCT = (
    "<3s 157s 18s B B B B i B i f f B B B B B H 56s 128s I h h H H "
    "f f f f H H 4s H H H B I H H H H f f 27s 4s"
)
ASD_FIELDS = (
    "co comments when program_version file_version itime dc_corr dc_time data_type "
    "ref_time ch1_wavel wavel_step data_format old_dc_count old_ref_count "
    "old_sample_count application channels app_data gps_data it fo dcc calibration "
    "instrument_num ymin ymax xmin xmax ip_numbits xmode flags dc_count ref_count "
    "sample_count instrument bulb swir1_gain swir2_gain swir1_offset swir2_offset "
    "splice1 splice2 smart_units spare"
).split()

HEADER_BYTES = 484

SPECTRA_TYPE = {
    0: "RAW", 1: "REFLECTANCE", 2: "RADIANCE", 3: "NOUNITS", 4: "IRRADIANCE",
    5: "QI", 6: "TRANSMITTANCE", 7: "UNKNOWN", 8: "ABSORBANCE",
}
DATA_FORMAT = {0: "<f4", 1: "<i4", 2: "<f8"}

DEFAULT_SPLICE1 = 1000.0
DEFAULT_SPLICE2 = 1800.0


def read_asd(path) -> dict:
    """Parse one ASD binary file.

    Returns a dict with ``spectrum`` and ``wavelength`` arrays plus the header
    metadata. Splice wavelengths are sanity-checked: many FieldSpec files leave
    those fields unwritten, so they come back as denormal floats, and the splice
    points are fixed in hardware anyway.
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw[:3] != b"ASD":
        raise ValueError(f"{path.name}: not an ASD binary file")
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path.name}: truncated, {len(raw)} bytes")

    m = dict(zip(ASD_FIELDS, struct.unpack_from(ASD_STRUCT, raw)))

    tm = struct.unpack("<9h", m["when"])  # C struct tm, nine shorts
    try:
        m["datetime"] = datetime.datetime(
            tm[5] + 1900, tm[4] + 1, tm[3], tm[2], tm[1], tm[0]
        )
    except ValueError:
        m["datetime"] = None

    n = m["channels"]
    dtype = DATA_FORMAT.get(m["data_format"], "<f4")
    expected = HEADER_BYTES + n * np.dtype(dtype).itemsize
    if len(raw) < expected:
        raise ValueError(f"{path.name}: expected {expected} bytes, found {len(raw)}")
    m["spectrum"] = np.frombuffer(raw, dtype, count=n, offset=HEADER_BYTES).astype(float)
    m["wavelength"] = m["ch1_wavel"] + np.arange(n) * m["wavel_step"]

    m["splice1"] = float(m["splice1"]) if 700 < m["splice1"] < 1400 else DEFAULT_SPLICE1
    m["splice2"] = float(m["splice2"]) if 1400 < m["splice2"] < 2200 else DEFAULT_SPLICE2

    m["type"] = SPECTRA_TYPE.get(m["data_type"], f"CODE_{m['data_type']}")
    m["is_raw_dn"] = m["data_type"] == 0
    m["file"] = path.name
    m["stem"] = path.stem
    suffix = path.suffix.lstrip(".")
    m["index"] = int(suffix) if suffix.isdigit() else -1
    return m


def is_asd_path(path) -> bool:
    """ASD files are named ``<stem>.NNN`` with a numeric extension."""
    path = Path(path)
    return path.is_file() and path.suffix.lstrip(".").isdigit()
