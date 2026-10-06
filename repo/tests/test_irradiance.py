import datetime

import numpy as np
import pytest

from asdspec import ModeledIrradiance, solar_zenith


@pytest.mark.parametrize("when,lat,lon,offset,expected", [
    # Solar noon at the equinox on the Greenwich meridian: the sun is a little past
    # the meridian because the equation of time is about -7.5 minutes in late March.
    (datetime.datetime(2024, 3, 20, 12, 0), 0.0, 0.0, 0, 1.9),
    # Northern winter solstice, local noon, mid-latitude: zenith angle is roughly
    # latitude plus the declination of 23.4 degrees.
    (datetime.datetime(2024, 12, 21, 12, 0), 40.76, -111.89, -7, 64.5),
])
def test_solar_zenith_against_known_geometry(when, lat, lon, offset, expected):
    assert solar_zenith(when, lat, lon, offset) == pytest.approx(expected, abs=0.6)


def test_minimum_daily_zenith_equals_latitude_minus_declination():
    """At the summer solstice the sun peaks at ``latitude - 23.44`` degrees."""
    lat, lon, offset = 40.76, -111.89, -6
    day = datetime.datetime(2024, 6, 21, 0, 0)
    angles = [solar_zenith(day + datetime.timedelta(minutes=m), lat, lon, offset)
              for m in range(0, 1440, 2)]
    assert min(angles) == pytest.approx(abs(lat - 23.44), abs=0.3)


def test_zenith_angle_is_symmetric_about_solar_noon():
    lat, lon, offset = 40.0, -111.0, -7
    noon = datetime.datetime(2024, 3, 20, 12, 0)
    # find local solar noon by minimising the zenith angle
    times = [noon + datetime.timedelta(minutes=m) for m in range(-120, 121)]
    angles = [solar_zenith(t, lat, lon, offset) for t in times]
    i = int(np.argmin(angles))
    before = solar_zenith(times[i] - datetime.timedelta(minutes=45), lat, lon, offset)
    after = solar_zenith(times[i] + datetime.timedelta(minutes=45), lat, lon, offset)
    assert before == pytest.approx(after, abs=0.3)


def _write_table(path, wavelengths_um, values_per_bin):
    lines = ["WVL,Z25,Z50"]
    for w, (a, b) in zip(wavelengths_um, values_per_bin):
        lines.append(f"{w},{a},{b}")
    lines += [",,", ",,"]  # trailing blank rows, as exported tables often carry
    path.write_text("\ufeff" + "\n".join(lines))
    return path


def test_band_integrated_table_is_converted_to_per_nm(tmp_path, wl):
    um = np.arange(0.31, 2.501, 0.01)
    # a triangular spectrum integrating to roughly 900 W m-2 when summed per band
    shape = np.exp(-((um - 0.5) / 0.6) ** 2)
    per_bin = 900 * shape / shape.sum()
    _write_table(tmp_path / "Irradiance.csv", np.round(um, 2),
                 list(zip(np.round(per_bin, 6), np.round(per_bin * 0.7, 6))))

    model = ModeledIrradiance.from_csv(tmp_path / "Irradiance.csv", wl)
    assert model.angles == [25.0, 50.0]
    assert "per 10 nm band" in model.unit_note
    assert model.totals[25.0] == pytest.approx(900, rel=0.05)
    # converted to spectral irradiance, so values are a tenth of the per-band ones
    peak = model.columns[25.0].max()
    assert peak == pytest.approx(per_bin.max() / 10, rel=0.05)


def test_nearest_column_and_clamping(tmp_path, wl):
    um = np.arange(0.31, 2.501, 0.01)
    shape = np.exp(-((um - 0.5) / 0.6) ** 2)
    per_bin = 900 * shape / shape.sum()
    _write_table(tmp_path / "Irradiance.csv", np.round(um, 2),
                 list(zip(np.round(per_bin, 6), np.round(per_bin * 0.7, 6))))
    model = ModeledIrradiance.from_csv(tmp_path / "Irradiance.csv", wl)

    _, angle, clamped = model.for_sza(30)
    assert (angle, clamped) == (25.0, False)
    _, angle, clamped = model.for_sza(21)
    assert (angle, clamped) == (25.0, True)
    _, angle, clamped = model.for_sza(60)
    assert (angle, clamped) == (50.0, True)
    assert model.for_sza(None)[0] is None


def test_table_without_recognisable_columns_raises(tmp_path, wl):
    path = tmp_path / "bad.csv"
    path.write_text("WVL,alpha,beta\n0.5,1,2\n0.6,1,2\n")
    with pytest.raises(ValueError, match="no zenith-angle columns"):
        ModeledIrradiance.from_csv(path, wl)
