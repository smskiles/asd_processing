"""Solar position and modeled clear-sky irradiance tables."""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd

TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def solar_zenith(local_time, latitude, longitude, utc_offset_hours):
    """Solar zenith angle in degrees, NOAA algorithm.

    ``local_time`` is naive local wall-clock time, which is what ASD headers
    store, so the UTC offset is required. Longitude is negative west.
    """
    dt = local_time - datetime.timedelta(hours=utc_offset_hours)
    year, month = dt.year, dt.month
    if month <= 2:
        year, month = year - 1, month + 12
    a = year // 100
    b = 2 - a + a // 4
    day = dt.day + (dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    jc = (jd - 2451545.0) / 36525.0

    mean_long = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eccent = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    centre = (np.sin(np.radians(mean_anom)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + np.sin(np.radians(2 * mean_anom)) * (0.019993 - 0.000101 * jc)
              + np.sin(np.radians(3 * mean_anom)) * 0.000289)
    true_long = mean_long + centre
    obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq += 0.00256 * np.cos(np.radians(125.04 - 1934.136 * jc))
    apparent = true_long - 0.00569 - 0.00478 * np.sin(np.radians(125.04 - 1934.136 * jc))
    declination = np.degrees(np.arcsin(np.sin(np.radians(obliq)) * np.sin(np.radians(apparent))))

    vy = np.tan(np.radians(obliq / 2)) ** 2
    eot = 4 * np.degrees(
        vy * np.sin(2 * np.radians(mean_long))
        - 2 * eccent * np.sin(np.radians(mean_anom))
        + 4 * eccent * vy * np.sin(np.radians(mean_anom)) * np.cos(2 * np.radians(mean_long))
        - 0.5 * vy * vy * np.sin(4 * np.radians(mean_long))
        - 1.25 * eccent * eccent * np.sin(2 * np.radians(mean_anom))
    )
    minutes = dt.hour * 60 + dt.minute + dt.second / 60
    true_solar_time = (minutes + eot + 4 * longitude) % 1440
    hour_angle = true_solar_time / 4 - 180
    cos_z = (np.sin(np.radians(latitude)) * np.sin(np.radians(declination))
             + np.cos(np.radians(latitude)) * np.cos(np.radians(declination))
             * np.cos(np.radians(hour_angle)))
    return float(np.degrees(np.arccos(np.clip(cos_z, -1, 1))))


class ModeledIrradiance:
    """Clear-sky spectra indexed by solar zenith angle, on a given wavelength grid.

    Expects a table whose first column is wavelength and whose remaining columns
    are named with a zenith angle, for example ``Z25`` or ``z50``. Wavelength in
    micrometres or nanometres is detected automatically, as is whether values are
    spectral irradiance per nanometre or integrated per wavelength bin: the two
    differ by the bin width, and only one of them integrates to a plausible
    broadband total.
    """

    PLAUSIBLE_TOTAL = (200.0, 1800.0)  # W m-2, clear sky, whole spectrum

    def __init__(self, columns, wavelength, unit_note="", totals=None, source=""):
        self.columns = columns
        self.wavelength = wavelength
        self.unit_note = unit_note
        self.totals = totals or {}
        self.source = source

    @classmethod
    def from_csv(cls, path, wavelength):
        path = Path(path)
        table = pd.read_csv(path, encoding="utf-8-sig")
        table.columns = [c.strip() for c in table.columns]
        table = table.dropna(how="all").dropna(subset=[table.columns[0]])

        raw_wl = pd.to_numeric(table[table.columns[0]], errors="coerce").values
        keep = np.isfinite(raw_wl)
        raw_wl, table = raw_wl[keep], table[keep]
        wl_nm = raw_wl * 1000 if np.nanmax(raw_wl) < 100 else raw_wl
        step = float(np.median(np.diff(wl_nm)))

        columns, totals, unit_note = {}, {}, ""
        for name in table.columns[1:]:
            match = re.search(r"(\d+(?:\.\d+)?)", str(name))
            if not match:
                continue
            values = pd.to_numeric(table[name], errors="coerce").values.astype(float)
            good = np.isfinite(values)
            if not good.any():
                continue
            total_per_nm = float(TRAPZ(values[good], wl_nm[good]))
            total_per_bin = float(np.nansum(values))
            lo, hi = cls.PLAUSIBLE_TOTAL
            if lo < total_per_bin < hi and not lo < total_per_nm < hi:
                values = values / step
                unit_note = f"W m-2 per {step:.0f} nm band"
            else:
                unit_note = "W m-2 nm-1"
            angle = float(match.group(1))
            columns[angle] = np.interp(wavelength, wl_nm, values)
            totals[angle] = float(TRAPZ(values, wl_nm))
        if not columns:
            raise ValueError(f"{path}: no zenith-angle columns recognised")
        return cls(columns, wavelength, unit_note, totals, source=path.name)

    @property
    def angles(self):
        return sorted(self.columns)

    def for_sza(self, sza):
        """Nearest column to a zenith angle.

        Returns ``(spectrum, angle, clamped)``. Nearest is sufficient: the weights
        used for broadband albedo are normalised, so only spectral shape enters,
        and shape changes slowly with zenith angle.
        """
        if sza is None:
            return None, None, False
        angle = min(self.columns, key=lambda a: abs(a - sza))
        clamped = not (min(self.columns) <= sza <= max(self.columns))
        return self.columns[angle], angle, clamped

    def describe(self):
        angles = ", ".join(f"{a:g} deg ({self.totals.get(a, float('nan')):.0f} W m-2)"
                           for a in self.angles)
        return (f"{self.source}: interpreted as {self.unit_note}; "
                f"zenith angles available: {angles}")
