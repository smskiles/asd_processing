import datetime

import numpy as np
import pytest

from asdspec import (broadband_albedo, compute_albedo, compute_reflectance, choose_panel,
                     find_blocks, panel_drift, scan_folder, splice_qc_table)
from test_dataset import build_albedo_day, build_transect_day


def test_albedo_recovers_a_known_spectrum(tmp_path, solar, snowy, wl):
    build_albedo_day(tmp_path, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    result = compute_albedo(data, blocks[0], blocks[1])

    good = ~result["masked"]
    np.testing.assert_allclose(result["albedo"][good], snowy[good], atol=0.02)
    assert result["it_factor"] == 1.0
    assert result["splice1_factor"] == pytest.approx(1.0, abs=5e-3)
    assert result["n_ref"] == result["n_tgt"] == 10


def test_integration_time_change_is_corrected(tmp_path, solar, snowy, wl):
    """A target taken at double the integration time must give the same albedo."""
    from conftest import write_asd
    t = datetime.datetime(2024, 5, 24, 14, 3, 0)
    for i in range(10):
        write_asd(tmp_path / f"d.{i:03d}", solar, t + datetime.timedelta(seconds=3 * i))
    t2 = t + datetime.timedelta(seconds=150)
    doubled = solar * snowy * np.where(wl <= 1000, 2.0, 1.0)
    for i in range(10):
        write_asd(tmp_path / f"d.{10 + i:03d}", doubled,
                  t2 + datetime.timedelta(seconds=3 * i), it_ms=136)

    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    result = compute_albedo(data, blocks[0], blocks[1])

    assert result["it_factor"] == pytest.approx(0.5)
    good = ~result["masked"]
    np.testing.assert_allclose(result["albedo"][good], snowy[good], atol=0.02)


def test_splice_qc_table_flags_a_bad_pair(tmp_path, solar, snowy, wl):
    """An uncorrected VNIR error shows up as a splice factor away from unity."""
    from conftest import write_asd
    t = datetime.datetime(2024, 5, 24, 14, 3, 0)
    for i in range(10):
        write_asd(tmp_path / f"d.{i:03d}", solar, t + datetime.timedelta(seconds=3 * i))
    t2 = t + datetime.timedelta(seconds=150)
    wrong = solar * snowy * np.where(wl <= 1000, 1.10, 1.0)
    for i in range(10):
        write_asd(tmp_path / f"d.{10 + i:03d}", wrong, t2 + datetime.timedelta(seconds=3 * i))

    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    result = compute_albedo(data, blocks[0], blocks[1])
    table = splice_qc_table([result])

    assert table.loc[0, "vnir_splice_factor"] == pytest.approx(1 / 1.10, rel=0.02)
    assert abs(table.loc[0, "raw_splice_step_pct"]) > 5


def test_broadband_albedo_of_a_flat_spectrum_is_that_value(wl):
    flat = np.full_like(wl, 0.42)
    irradiance = np.exp(-((wl - 600) / 700) ** 2)
    out = broadband_albedo(wl, flat, irradiance)
    for value in out.values():
        assert value == pytest.approx(0.42)


def test_broadband_albedo_is_weighted_not_a_plain_mean(wl, snowy):
    visible_heavy = np.where(wl < 700, 1.0, 0.05)
    swir_heavy = np.where(wl < 700, 0.05, 1.0)
    assert (broadband_albedo(wl, snowy, visible_heavy)["broadband 350-2500"]
            > broadband_albedo(wl, snowy, swir_heavy)["broadband 350-2500"])


def test_reflectance_keeps_points_separate(tmp_path, solar, snowy):
    build_transect_day(tmp_path, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    target = [b for b in blocks.values() if b["role"] == "target"][0]
    panel, reason = choose_panel(blocks, "260829r5", target)

    assert panel["block_id"] == 0
    assert "settings match" in reason

    result = compute_reflectance(data, panel, target, panel_reflectance=0.99)
    assert result["reflectance"].shape[0] == 21
    assert result["n_panel"] == 10
    assert result["it_factor"] == pytest.approx(0.5)
    # the synthetic transect brightens monotonically along its length
    levels = np.nanmean(result["reflectance"][:, :300], axis=1)
    assert np.all(np.diff(levels) > 0)


def test_panel_drift_reports_a_settings_change(tmp_path, solar, snowy):
    build_transect_day(tmp_path, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    ratio, same_settings, minutes = panel_drift(data, blocks, "260829r5")
    assert not same_settings
    assert minutes == pytest.approx(4.3, abs=0.5)
    assert np.isfinite(ratio).any()
