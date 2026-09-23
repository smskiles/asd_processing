"""Spectral and broadband albedo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .corrections import (DEFAULT_MASK, DEFAULT_SNR_MIN, integration_time_scale,
                          mask_fill, snr_mask, splice_correct, splice_step)
from .irradiance import TRAPZ

CLIP_RANGE = (0.0, 1.05)

BANDS = {
    "broadband 350-2500": (350, 2500),
    "VIS 350-700": (350, 700),
    "NIR 700-2500": (700, 2500),
    "SWIR 1000-2500": (1000, 2500),
}


def compute_albedo(dataset, ref_block, tgt_block, exclude=None, manual_it_factor=None,
                   splice_mode="swir1_anchor", correct_swir2=False,
                   mask_windows=DEFAULT_MASK, snr_min=DEFAULT_SNR_MIN,
                   clip=CLIP_RANGE):
    """Albedo as the averaged target set over the averaged reference set.

    Applies the integration-time correction, then the splice correction, then
    masking. The returned dict keeps the intermediate spectra so the effect of
    each step can be inspected.
    """
    exclude = exclude or {}
    wl = dataset.wavelength
    ref, kept_ref = dataset.mean(ref_block, exclude.get(ref_block["block_id"], ()))
    tgt, kept_tgt = dataset.mean(tgt_block, exclude.get(tgt_block["block_id"], ()))

    scale, it_factor, notes = integration_time_scale(wl, tgt_block, ref_block, manual_it_factor)
    with np.errstate(divide="ignore", invalid="ignore"):
        after_it = (tgt / ref) * scale
    after_it[~np.isfinite(after_it)] = np.nan

    after_splice, f1, f2, splice_notes = splice_correct(
        wl, after_it, ref_block["splice1"], ref_block["splice2"], splice_mode, correct_swir2
    )
    notes += splice_notes

    extra = snr_mask(dataset.stack(tgt_block), snr_min) | snr_mask(dataset.stack(ref_block), snr_min)
    albedo, masked = mask_fill(wl, after_splice, mask_windows, extra)
    n_clipped = int(((albedo < clip[0]) | (albedo > clip[1])).sum())
    albedo = np.clip(albedo, *clip)
    notes.append(f"masked {masked.sum()} of {len(wl)} channels; clipped {n_clipped}")

    return dict(
        label=f"{tgt_block['stem']} b{ref_block['block_id']}/b{tgt_block['block_id']}",
        wavelength=wl, albedo=albedo, albedo_after_it=after_it, albedo_after_splice=after_splice,
        masked=masked, ref_block=ref_block["block_id"], tgt_block=tgt_block["block_id"],
        it_factor=it_factor, splice1_factor=f1, splice2_factor=f2,
        raw_splice_step=splice_step(wl, after_it, ref_block["splice1"]),
        n_ref=len(kept_ref), n_tgt=len(kept_tgt), notes=notes, time=tgt_block["t_start"],
    )


def broadband_albedo(wl, albedo, irradiance, bands=BANDS):
    """Irradiance-weighted mean albedo over each band.

    The weighting spectrum must be in physical units. Raw uplooking DN will not
    do: DN is irradiance times the instrument's spectral responsivity times
    integration time, and that responsivity is strongly wavelength dependent, so
    weighting by DN silently reweights the integral.
    """
    out = {}
    for name, (lo, hi) in bands.items():
        m = ((wl >= lo) & (wl <= hi)
             & np.isfinite(albedo) & np.isfinite(irradiance))
        denominator = TRAPZ(irradiance[m], wl[m])
        out[name] = TRAPZ(albedo[m] * irradiance[m], wl[m]) / denominator if denominator > 0 else np.nan
    return out


def splice_qc_table(results) -> pd.DataFrame:
    """Per-set QC. The VNIR splice factor is the column to read.

    After a correct integration-time correction it sits near 0.985. A value well
    outside that flags a pair where the instrument was re-optimized between
    reference and target, so the integration-time factor is suspect. The two
    corrections both act on the VNIR and cannot be separated after the fact.
    """
    rows = []
    for r in results:
        rows.append(dict(
            set=r["label"],
            time=r["time"].strftime("%Y-%m-%d %H:%M") if r["time"] else "",
            n_ref=r["n_ref"], n_tgt=r["n_tgt"],
            it_factor=round(r["it_factor"], 4),
            raw_splice_step_pct=round(100 * (r["raw_splice_step"] - 1), 1),
            vnir_splice_factor=round(r["splice1_factor"], 4),
        ))
    return pd.DataFrame(rows)
