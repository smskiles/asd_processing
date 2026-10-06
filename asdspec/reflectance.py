"""Panel-referenced reflectance along a transect."""

from __future__ import annotations

import numpy as np

from .corrections import (implied_vnir_offset, integration_time_scale,
                          splice_correct, subtract_vnir_offset)


def choose_panel(blocks, stem, target_block, override=None):
    """Pick the white reference block to use for a transect block.

    Prefers a panel whose gain and offset match the transect, since a panel taken
    after a re-optimization cannot be used for the SWIR. Falls back to nearest in
    time. Returns ``(block, reason)``.
    """
    if override is not None:
        return blocks[override], "manual override"
    refs = [b for b in blocks.values() if b["stem"] == stem and b["role"] == "reference"]
    if not refs:
        raise ValueError(f"no reference block found for stem {stem}")
    settings = lambda b: (b["swir1_gain"], b["swir1_offset"], b["swir2_gain"], b["swir2_offset"])
    matching = [b for b in refs if settings(b) == settings(target_block)]
    pool = matching or refs
    reason = "settings match the transect" if matching else "no settings match, nearest in time"
    chosen = min(pool, key=lambda b: abs((b["t_start"] - target_block["t_start"]).total_seconds())
                 if b["t_start"] and target_block["t_start"] else 0)
    return chosen, reason


def panel_drift(dataset, blocks, stem):
    """Closing-over-opening panel ratio, as a stability check on the transect.

    If the instrument was re-optimized before the closing panel, the ratio mixes
    illumination drift with the settings change and only the VNIR is
    interpretable. Returns ``(ratio, same_settings, minutes)`` or ``None``.
    """
    refs = sorted([b for b in blocks.values() if b["stem"] == stem and b["role"] == "reference"],
                  key=lambda b: b["t_start"])
    if len(refs) < 2:
        return None
    first, _ = dataset.mean(refs[0])
    last, _ = dataset.mean(refs[-1])
    settings = lambda b: (b["swir1_gain"], b["swir1_offset"], b["swir2_gain"], b["swir2_offset"])
    minutes = (refs[-1]["t_start"] - refs[0]["t_start"]).total_seconds() / 60
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = last / first
    return ratio, settings(refs[0]) == settings(refs[-1]), minutes


def compute_reflectance(dataset, panel_block, target_block, panel_reflectance=0.99,
                        exclude=None, manual_it_factor=None, splice_mode="offset",
                        correct_swir2=False):
    """Reflectance of every point in a transect against one averaged panel.

    Transect points are not averaged, since the variability along the transect is
    the signal. The splice factor is derived once from the block mean and applied
    to every point, rather than solved per point, which would be unstable on the
    darkest targets.

    With a narrow foreoptic under natural illumination this is an HDRF, not a
    bi-hemispherical albedo. Over snow and ice, where the forward scattering lobe
    is strong, the two are not interchangeable.
    """
    exclude = exclude or {}
    wl = dataset.wavelength
    panel, kept_panel = dataset.mean(panel_block, exclude.get(panel_block["block_id"], ()))
    target_mean, _ = dataset.mean(target_block, exclude.get(target_block["block_id"], ()))
    panel_r = (np.full_like(wl, float(panel_reflectance))
               if np.isscalar(panel_reflectance) else np.asarray(panel_reflectance, dtype=float))

    scale, it_factor, notes = integration_time_scale(wl, target_block, panel_block, manual_it_factor)
    is_raw = panel_block["type"] == "RAW"

    # The offset is derived once from the block mean, since a single transect
    # point is too noisy near 1000 nm to solve for it reliably.
    offset, panel_corrected, segment = np.nan, panel, np.ones_like(wl)
    if splice_mode == "offset" and is_raw:
        _, panel_corrected, offset, splice_notes = subtract_vnir_offset(
            wl, target_mean, panel, it_factor, panel_block["splice1"])
        notes += splice_notes
        f1 = f2 = 1.0
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_reflectance = (target_mean / panel) * scale * panel_r
        mean_reflectance[~np.isfinite(mean_reflectance)] = np.nan
        mode = "swir1_anchor" if splice_mode == "offset" else splice_mode
        _, f1, f2, splice_notes = splice_correct(
            wl, mean_reflectance, panel_block["splice1"], panel_block["splice2"],
            mode, correct_swir2)
        notes += splice_notes
        if is_raw:
            offset = implied_vnir_offset(wl, target_mean, panel, it_factor,
                                         panel_block["splice1"])
        if mode == "swir1_anchor":
            segment = np.where(wl <= panel_block["splice1"], f1, 1.0)
        elif mode == "vnir_anchor":
            segment = np.where(wl > panel_block["splice1"], f1, 1.0)
        if correct_swir2:
            segment = segment * np.where(wl > panel_block["splice2"], f2, 1.0)

    vnir = wl <= panel_block["splice1"]
    drop = set(exclude.get(target_block["block_id"], ()))
    spectra, labels = [], []
    for path, index in zip(target_block["paths"], target_block["indices"]):
        if index in drop:
            continue
        point = dataset.spectra[path].astype(float).copy()
        if splice_mode == "offset" and is_raw and np.isfinite(offset) and offset > 0:
            point[vnir] -= offset
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (point / panel_corrected) * scale * panel_r * segment
        r[~np.isfinite(r)] = np.nan
        spectra.append(r)
        labels.append(index)

    return dict(
        label=f"{target_block['stem']} b{target_block['block_id']}",
        wavelength=wl, reflectance=np.array(spectra), labels=labels,
        panel_block=panel_block["block_id"], target_block=target_block["block_id"],
        n_panel=len(kept_panel), it_factor=it_factor, splice1_factor=f1, splice2_factor=f2,
        vnir_offset_dn=offset,
        notes=notes, time=target_block["t_start"],
    )
