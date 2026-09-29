"""Outlier scoring for replicate measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd

QC_BAND = (450, 1300)
Z_THRESHOLD = 3.5


def outlier_scores(stack, wl, qc_band=QC_BAND, z_threshold=Z_THRESHOLD) -> pd.DataFrame:
    """Score each replicate on overall level and on spectral shape.

    Both scores are modified z-scores built on the median absolute deviation
    rather than the standard deviation. With ten replicates, one bad spectrum
    inflates a standard deviation enough to hide itself.

    Level catches a scan taken while a cloud crossed, or a saturated one. Shape,
    computed after dividing each spectrum by its own mean, catches a shadow on
    the receptor, a tipped sensor, or an obstruction in the field of view.

    A steady drift in level across a reference set is normal as the sun moves.
    Flags are a prompt to look, not a verdict.
    """
    stack = np.asarray(stack, dtype=float)
    band = (wl >= qc_band[0]) & (wl <= qc_band[1])
    x = stack[:, band]

    level = np.nanmean(x, axis=1)
    med = np.median(level)
    mad = np.median(np.abs(level - med))
    z_level = 0.6745 * (level - med) / mad if mad > 0 else np.zeros_like(level)

    normalised = x / level[:, None]
    shape_median = np.median(normalised, axis=0)
    resid = np.sqrt(np.nanmean((normalised - shape_median) ** 2, axis=1))
    med_s = np.median(resid)
    mad_s = np.median(np.abs(resid - med_s))
    z_shape = 0.6745 * (resid - med_s) / mad_s if mad_s > 0 else np.zeros_like(resid)

    return pd.DataFrame(dict(
        level=level, z_level=z_level, shape_rmse=resid, z_shape=z_shape,
        flagged=(np.abs(z_level) > z_threshold) | (z_shape > z_threshold),
    ))


def qc_block(dataset, block, exclude=(), z_threshold=Z_THRESHOLD) -> pd.DataFrame:
    """Outlier scores for one block, annotated with file names and exclusions."""
    scores = outlier_scores(dataset.stack(block), dataset.wavelength, z_threshold=z_threshold)
    scores.insert(0, "index", block["indices"])
    scores.insert(1, "file", block["files"])
    scores["excluded"] = scores["index"].isin(set(exclude))
    return scores


def qc_all(dataset, blocks, exclude=None, z_threshold=Z_THRESHOLD) -> dict:
    exclude = exclude or {}
    return {
        bid: qc_block(dataset, b, exclude.get(bid, ()), z_threshold)
        for bid, b in blocks.items()
    }
