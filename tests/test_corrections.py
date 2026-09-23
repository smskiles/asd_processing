import numpy as np
import pytest

from asdspec import integration_time_scale, mask_fill, snr_mask, splice_correct, splice_step


def block(it_ms=68, type_="RAW", gain=4096, offset=1544):
    return dict(it_ms=it_ms, type=type_, splice1=1000.0, splice2=1800.0,
                swir1_gain=gain, swir1_offset=offset, swir2_gain=2816, swir2_offset=1024)


def test_matched_integration_time_is_a_no_op(wl):
    scale, factor, notes = integration_time_scale(wl, block(68), block(68))
    assert factor == 1.0
    np.testing.assert_allclose(scale, 1.0)
    assert notes == []


def test_scaling_applies_below_the_splice_only(wl):
    scale, factor, _ = integration_time_scale(wl, block(136), block(68))
    assert factor == pytest.approx(0.5)
    assert scale[wl <= 1000].min() == pytest.approx(0.5)
    assert scale[wl > 1000].max() == pytest.approx(1.0)


def test_calibrated_files_are_not_scaled(wl):
    scale, factor, notes = integration_time_scale(wl, block(136, "IRRADIANCE"),
                                                  block(68, "IRRADIANCE"))
    assert factor == 1.0
    np.testing.assert_allclose(scale, 1.0)
    assert "no integration-time scaling" in notes[0]


def test_mismatched_swir_settings_warn(wl):
    _, _, notes = integration_time_scale(wl, block(68, offset=1544), block(68, offset=1800))
    assert any("SWIR gain/offset differ" in n for n in notes)


def test_splice_step_ignores_real_spectral_slope(wl, snowy):
    """A continuous spectrum with a steep slope must report no step."""
    assert splice_step(wl, snowy, 1000.0) == pytest.approx(1.0, abs=2e-3)


def test_splice_step_recovers_a_known_step(wl, snowy):
    y = snowy.copy()
    y[wl <= 1000] /= 0.94
    assert splice_step(wl, y, 1000.0) == pytest.approx(0.94, rel=5e-3)


def test_splice_correct_restores_continuity(wl, snowy):
    y = snowy.copy()
    y[wl <= 1000] /= 0.94
    corrected, f1, _, _ = splice_correct(wl, y, mode="swir1_anchor")
    assert f1 == pytest.approx(0.94, rel=5e-3)
    np.testing.assert_allclose(corrected, snowy, rtol=1e-2)
    assert splice_step(wl, corrected, 1000.0) == pytest.approx(1.0, abs=5e-3)


def test_swir1_anchor_leaves_the_swir_untouched(wl, snowy):
    y = snowy.copy()
    y[wl <= 1000] /= 0.94
    corrected, _, _, _ = splice_correct(wl, y, mode="swir1_anchor")
    np.testing.assert_allclose(corrected[wl > 1000], y[wl > 1000])


def test_vnir_anchor_leaves_the_vnir_untouched(wl, snowy):
    y = snowy.copy()
    y[wl <= 1000] /= 0.94
    corrected, _, _, _ = splice_correct(wl, y, mode="vnir_anchor")
    np.testing.assert_allclose(corrected[wl <= 1000], y[wl <= 1000])


def test_splice_mode_none_and_unknown(wl, snowy):
    same, f1, f2, _ = splice_correct(wl, snowy, mode="none")
    np.testing.assert_allclose(same, snowy)
    assert (f1, f2) == (1.0, 1.0)
    with pytest.raises(ValueError):
        splice_correct(wl, snowy, mode="nonsense")


def test_implausible_splice_factor_is_refused(wl, snowy):
    y = snowy.copy()
    y[wl <= 1000] *= 10.0
    _, f1, _, notes = splice_correct(wl, y, mode="swir1_anchor")
    assert f1 == 1.0
    assert any("implausible" in n for n in notes)


def test_snr_mask_flags_noise_not_signal(wl):
    rng = np.random.default_rng(0)
    clean = np.tile(np.full_like(wl, 100.0), (10, 1)) + rng.normal(0, 0.1, (10, len(wl)))
    noisy = clean.copy()
    noisy[:, wl > 2000] = rng.normal(0, 50, noisy[:, wl > 2000].shape)
    mask = snr_mask(noisy, snr_min=5.0)
    assert not mask[wl < 1000].any()
    assert mask[wl > 2000].mean() > 0.8
    assert not snr_mask(noisy, snr_min=0).any()


def test_mask_fill_blanks_and_interpolates(wl):
    y = np.ones_like(wl)
    filled, mask = mask_fill(wl, y, windows=[(1350, 1450)])
    assert mask[(wl >= 1350) & (wl <= 1450)].all()
    assert np.isfinite(filled).all()
    np.testing.assert_allclose(filled, 1.0)
    blanked, _ = mask_fill(wl, y, windows=[(1350, 1450)], fill=False)
    assert np.isnan(blanked[(wl >= 1350) & (wl <= 1450)]).all()
