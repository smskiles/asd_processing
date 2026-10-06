"""Integration time, detector splice, and noise masking."""

from __future__ import annotations

import numpy as np

DEFAULT_MASK = [(350, 400), (1350, 1450), (1800, 1950), (2350, 2500)]
DEFAULT_SNR_MIN = 5.0
DEFAULT_SPLICE_SPAN = 10.0
MAX_SPLICE_FACTOR = 1.5

#: Expected VNIR splice factor once the integration-time correction is right.
#: Measured on a FieldSpec 4 from pairs sharing an integration time.
NOMINAL_SPLICE_FACTOR = 0.985
SPLICE_FACTOR_OK = (0.96, 1.01)


def integration_time_scale(wl, num_block, den_block, manual=None):
    """VNIR scale factor for the ratio ``num / den``.

    ASD raw DN in the VNIR scales linearly with integration time. The SWIR uses
    gain and offset instead and does not respond to it, so the correction applies
    only below the first splice, and only to raw DN files.

    Returns ``(scale_array, factor, notes)``.
    """
    notes = []
    if num_block["type"] != "RAW" or den_block["type"] != "RAW":
        return np.ones_like(wl), 1.0, ["calibrated files, no integration-time scaling"]

    splice1 = den_block["splice1"]
    if num_block["it_ms"] == den_block["it_ms"]:
        factor = 1.0
    elif manual:
        factor = float(manual)
        notes.append(f"manual integration-time factor {factor:.4f}")
    else:
        factor = den_block["it_ms"] / num_block["it_ms"]
        notes.append(
            f"integration time differs ({num_block['it_ms']} vs {den_block['it_ms']} ms): "
            f"VNIR scaled by {factor:.4f}"
        )

    settings = lambda b: (b["swir1_gain"], b["swir1_offset"], b["swir2_gain"], b["swir2_offset"])
    if settings(num_block) != settings(den_block):
        notes.append(
            "WARNING: SWIR gain/offset differ between blocks. The SWIR ratio is not "
            "reliable and cannot be corrected after the fact."
        )
    return np.where(wl <= splice1, factor, 1.0), factor, notes


def _edge_value(wl, y, boundary, span, side):
    """Linear fit on one side of a boundary, extrapolated to the boundary."""
    if side == "lo":
        m = (wl >= boundary - span) & (wl <= boundary)
    else:
        m = (wl >= boundary + 1) & (wl <= boundary + 1 + span)
    if m.sum() < 3 or not np.isfinite(y[m]).all():
        return np.nan
    return float(np.polyval(np.polyfit(wl[m], y[m], 1), boundary + 0.5))


def splice_step(wl, y, boundary, span=DEFAULT_SPLICE_SPAN):
    """Ratio of the SWIR-side value to the VNIR-side value at a boundary.

    Both sides are fitted locally and extrapolated to the boundary, which keeps
    the real spectral slope out of the estimate. Comparing window means either
    side instead would bake the slope in: over snow near 1000 nm that alone
    accounts for roughly three quarters of the apparent step.
    """
    lo = _edge_value(wl, y, boundary, span, "lo")
    hi = _edge_value(wl, y, boundary, span, "hi")
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo == 0:
        return np.nan
    return hi / lo


def subtract_vnir_offset(wl, numerator, denominator, it_factor=1.0, splice=1000.0,
                         max_fraction=0.5):
    """Remove the additive VNIR offset from both spectra before they are ratioed.

    This is the physically right way to close the step at 1000 nm. The offset is
    a roughly constant number of DN, so the reflectance error it causes depends
    on how large the signal is: negligible in the visible, where a panel reads
    tens of thousands of DN, and several percent near 1000 nm, where the silicon
    response has collapsed to around a thousand. Subtracting it in DN removes the
    error where it exists and leaves the rest of the VNIR alone.

    Scaling the ratio instead, as ``splice_correct`` does, spreads a correction
    sized for the splice across the whole VNIR. On a dark target that turns a
    reflectance error of a few tenths of a percent at 500 nm into one of eight or
    nine percent, in the part of the spectrum where light-absorbing particles are
    measured.

    Takes and returns raw DN. Returns ``(numerator, denominator, offset, notes)``.
    """
    wl = np.asarray(wl, dtype=float)
    numerator = np.asarray(numerator, dtype=float).copy()
    denominator = np.asarray(denominator, dtype=float).copy()

    offset = implied_vnir_offset(wl, numerator, denominator, it_factor, splice)
    if not np.isfinite(offset):
        return numerator, denominator, np.nan, ["splice: offset not estimable, left alone"]
    if offset <= 0:
        return (numerator, denominator, offset,
                [f"splice: implied offset {offset:.0f} DN is not positive, so an additive "
                 f"offset does not explain this step; left alone"])

    vnir = wl <= splice
    floor = float(min(np.nanmin(numerator[vnir]), np.nanmin(denominator[vnir])))
    if offset > max_fraction * floor:
        return (numerator, denominator, offset,
                [f"splice: implied offset {offset:.0f} DN exceeds half the smallest VNIR "
                 f"signal ({floor:.0f} DN), so subtracting it is not safe; left alone"])

    numerator[vnir] -= offset
    denominator[vnir] -= offset
    notes = [f"splice: subtracted a {offset:.0f} DN VNIR offset from both spectra"]
    if not 0.25 * TYPICAL_VNIR_OFFSET_DN <= offset <= 3 * TYPICAL_VNIR_OFFSET_DN:
        notes.append(f"   WARNING: {offset:.0f} DN is far from the {TYPICAL_VNIR_OFFSET_DN:.0f} "
                     f"DN an instrument offset usually implies, so the step is probably not an "
                     f"offset at all. An unaccounted integration time is the usual cause; the "
                     f"subtraction will still close the step but is hiding that rather than "
                     f"fixing it.")
    return numerator, denominator, offset, notes


def splice_correct(wl, y, splice1=1000.0, splice2=1800.0, mode="swir1_anchor",
                   correct_swir2=False, span=DEFAULT_SPLICE_SPAN):
    """Scale detector segments so the spectrum is continuous at the boundaries.

    ``mode="swir1_anchor"`` scales the VNIR to meet SWIR1 and leaves SWIR1 alone,
    matching what the instrument software does. ``"vnir_anchor"`` does the
    reverse. ``"none"`` returns the input unchanged.

    The 1800 nm boundary is off by default: over snow it sits in a low-signal,
    water-vapour-affected region where the factor is poorly constrained.

    Returns ``(corrected, factor1, factor2, notes)``.
    """
    y = np.asarray(y, dtype=float).copy()
    f1 = f2 = 1.0
    notes = []
    if mode == "none":
        return y, f1, f2, ["splice correction off"]
    if mode not in ("swir1_anchor", "vnir_anchor"):
        raise ValueError(f"unknown splice mode {mode!r}")

    step = splice_step(wl, y, splice1, span)
    if np.isfinite(step) and step != 0:
        f1 = step if mode == "swir1_anchor" else 1.0 / step
        if abs(np.log(f1)) < np.log(MAX_SPLICE_FACTOR):
            segment = wl <= splice1 if mode == "swir1_anchor" else wl > splice1
            y[segment] *= f1
            notes.append(
                f"{splice1:.0f} nm: raw step {100 * (step - 1):+.1f} %, "
                f"{'VNIR' if mode == 'swir1_anchor' else 'SWIR'} scaled by {f1:.4f}"
            )
        else:
            notes.append(f"{splice1:.0f} nm: factor {f1:.3f} implausible, not applied")
            f1 = 1.0
    else:
        notes.append(f"{splice1:.0f} nm: not enough valid signal, no correction")

    if correct_swir2:
        step2 = splice_step(wl, y, splice2, span)
        if np.isfinite(step2) and step2 != 0:
            f2 = 1.0 / step2
            if abs(np.log(abs(f2))) < np.log(MAX_SPLICE_FACTOR):
                y[wl > splice2] *= f2
                notes.append(f"{splice2:.0f} nm: SWIR2 scaled by {f2:.4f}")
            else:
                notes.append(f"{splice2:.0f} nm: factor {f2:.3f} implausible, not applied")
                f2 = 1.0
    return y, f1, f2, notes


#: A small additive offset in the VNIR, in DN, consistent with the splice step
#: seen on a FieldSpec 4 across bright snow and dark glacier ice.
TYPICAL_VNIR_OFFSET_DN = 20.0


def implied_vnir_offset(wl, numerator, denominator, it_factor=1.0, splice=1000.0):
    """Additive VNIR offset, in DN, that would explain the step at the splice.

    Near 1000 nm the silicon detector response has collapsed, so the raw DN is
    small and any additive offset, residual dark current or stray light, is a
    large fraction of it. That offset inflates a ratio whose numerator is much
    smaller than its denominator, which is why the step grows as the target gets
    darker: bright snow shows about 1.5 percent, dark glacier ice about 8.

    Solving for the offset turns the step into a number that should be roughly
    constant for an instrument, around 20 DN on a FieldSpec 4, whatever the
    target. A value near that means the step is ordinary and the splice
    correction is doing its job. A very different one means something else is
    wrong, most often an integration time that was not accounted for.

    Takes the two averaged spectra in raw DN, not a ratio.
    """
    wl = np.asarray(wl, dtype=float)
    lo = int(np.argmin(np.abs(wl - splice)))
    hi = lo + 1
    if hi >= wl.size:
        return np.nan
    num_v, den_v = float(numerator[lo]), float(denominator[lo])
    num_s, den_s = float(numerator[hi]), float(denominator[hi])
    if den_v == 0 or den_s == 0:
        return np.nan
    swir = num_s / den_s
    if swir == it_factor:
        return np.nan
    return (swir * den_v - it_factor * num_v) / (swir - it_factor)


def snr_mask(stack, snr_min=DEFAULT_SNR_MIN):
    """Channels where replicate scatter is large relative to the signal."""
    stack = np.asarray(stack, dtype=float)
    if snr_min <= 0:
        return np.zeros(stack.shape[1], dtype=bool)
    mu = np.nanmean(stack, axis=0)
    sd = np.nanstd(stack, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(mu) / sd
    return (~np.isfinite(snr)) | (snr < snr_min)


def mask_fill(wl, y, windows=DEFAULT_MASK, extra_mask=None, fill=True):
    """Blank fixed windows plus any extra mask, optionally interpolating across.

    Returns ``(values, mask)``. Masked channels are interpolated when ``fill`` is
    true so the result can be integrated; the mask is returned so plots can leave
    them blank.
    """
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)
    bad = np.zeros_like(wl, dtype=bool)
    for lo, hi in windows:
        bad |= (wl >= lo) & (wl <= hi)
    if extra_mask is not None:
        bad |= np.asarray(extra_mask, dtype=bool)
    bad |= ~np.isfinite(y)

    out = y.copy()
    out[bad] = np.nan
    if fill and (~bad).any():
        out = np.interp(wl, wl[~bad], y[~bad])
    return out, bad
