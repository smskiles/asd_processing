"""Plotting helpers. All take explicit data and return the figure."""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt

from .corrections import mask_fill
from .irradiance import TRAPZ


def plot_block_qc(dataset, block, scores, z_threshold=3.5):
    """Replicates, their departure from the block median, and the level score."""
    wl = dataset.wavelength
    stack = dataset.stack(block)
    median = np.median(stack, axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), gridspec_kw={"width_ratios": [2, 2, 1.3]})
    for i, row in scores.iterrows():
        bad = bool(row["flagged"] or row["excluded"])
        style = dict(lw=1.4 if bad else 0.7, color="crimson" if bad else "steelblue",
                     ls="--" if bad else "-", alpha=0.95 if bad else 0.65)
        ax[0].plot(wl, stack[i], label=f"{row['index']:03d}" if bad else None, **style)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax[1].plot(wl, stack[i] / median, **style)
    ax[0].plot(wl, median, color="k", lw=1.4, label="median")
    ax[0].set(xlabel="wavelength (nm)",
              ylabel="DN" if block["type"] == "RAW" else block["type"].lower(),
              title=f"block {block['block_id']}  {block['stem']}  {block['role']}  "
                    f"n={block['n']}  it={block['it_ms']} ms")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].axhline(1, color="k", lw=1)
    ax[1].set(xlabel="wavelength (nm)", ylabel="ratio to block median", ylim=(0.9, 1.1),
              title="shape and level departure")

    y = np.arange(len(scores))
    ax[2].barh(y, scores["z_level"],
               color=["crimson" if f else "steelblue" for f in scores["flagged"]])
    for sign in (-1, 1):
        ax[2].axvline(sign * z_threshold, color="crimson", ls=":", lw=1)
    ax[2].set(yticks=y, yticklabels=[f"{i:03d}" for i in scores["index"]],
              xlabel="z (level)", title="outlier score")
    ax[2].invert_yaxis()
    fig.tight_layout()
    return fig


def plot_albedo(results, splice_mode="", max_legend=30):
    """Spectral albedo for every set, with a VNIR detail panel."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    colours = plt.cm.viridis(np.linspace(0, 0.95, len(results)))
    for k, r in enumerate(results):
        y = np.where(r["masked"], np.nan, r["albedo"])
        label = r["label"] + (f"  {r['time']:%m-%d %H:%M}" if r["time"] else "")
        ax[0].plot(r["wavelength"], y, lw=1.0, color=colours[k], label=label)
        ax[1].plot(r["wavelength"], y, lw=1.0, color=colours[k])
    ax[0].set(xlabel="wavelength (nm)", ylabel="albedo", ylim=(0, 1.05),
              title=f"spectral albedo  (splice: {splice_mode})" if splice_mode
              else "spectral albedo")
    for boundary in (1000, 1800):
        ax[1].axvline(boundary, color="grey", ls=":", lw=1)
    ax[1].set(xlabel="wavelength (nm)", ylabel="albedo", xlim=(350, 1400), ylim=(0, 1.05),
              title="VNIR detail, dotted line = 1000 nm splice")
    if len(results) <= max_legend:
        fig.legend(*ax[0].get_legend_handles_labels(), loc="center left",
                   bbox_to_anchor=(1.0, 0.5), fontsize=6, frameon=False)
    fig.tight_layout()
    return fig


def plot_splice_diagnostic(results, boundary=1000.0, window=100.0, normalise_at=980.0):
    """Before and after the splice correction, normalised so sets are comparable."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    colours = plt.cm.viridis(np.linspace(0, 0.95, len(results)))
    for k, r in enumerate(results):
        wl = r["wavelength"]
        zoom = (wl >= boundary - window) & (wl <= boundary + window)
        ref = int(np.argmin(np.abs(wl - normalise_at)))
        for axis, key in zip(ax, ("albedo_after_it", "albedo_after_splice")):
            y = r[key]
            if np.isfinite(y[ref]) and y[ref] != 0:
                axis.plot(wl[zoom], y[zoom] / y[ref], lw=0.9, color=colours[k])
    for axis, title in zip(ax, ("before splice correction", "after splice correction")):
        axis.axvline(boundary, color="crimson", ls=":", lw=1.2)
        axis.set(xlabel="wavelength (nm)", title=title)
    ax[0].set_ylabel(f"albedo, normalised at {normalise_at:.0f} nm")
    fig.tight_layout()
    return fig


def plot_irradiance_comparison(wl, measured, modeled, modeled_label="modeled"):
    """Measured against modeled irradiance, absolute and normalised."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    ax[0].plot(wl, measured, lw=1.1, label="measured")
    ax[0].plot(wl, modeled, lw=1.1, label=modeled_label)
    ax[0].set(xlabel="wavelength (nm)", ylabel="W m$^{-2}$ nm$^{-1}$", title="absolute")
    ax[0].legend(fontsize=8)
    ax[1].plot(wl, measured / TRAPZ(measured, wl), lw=1.1, label="measured")
    ax[1].plot(wl, modeled / TRAPZ(modeled, wl), lw=1.1, label=modeled_label)
    ax[1].set(xlabel="wavelength (nm)", ylabel="normalised weight",
              title="shape, which is all that enters the weighted mean")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_reflectance(result, mask_windows=None):
    """Per-point transect reflectance and the spread across the transect."""
    wl = result["wavelength"]
    R = result["reflectance"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    colours = plt.cm.viridis(np.linspace(0, 1, len(result["labels"])))
    for j, (spectrum, label) in enumerate(zip(R, result["labels"])):
        y = (mask_fill(wl, spectrum, mask_windows, fill=False)[0]
             if mask_windows is not None else spectrum)
        ax[0].plot(wl, y, lw=0.9, color=colours[j], label=f"{label:03d}")
    low, high = np.nanpercentile(R, [10, 90], axis=0)
    ax[1].fill_between(wl, low, high, alpha=0.3, color="steelblue", label="10th to 90th pct")
    ax[1].plot(wl, np.nanmedian(R, axis=0), color="k", lw=1.3, label="median")
    for axis in ax:
        axis.set(xlabel="wavelength (nm)", ylabel="reflectance (HDRF)", ylim=(0, 1.05))
    ax[0].set_title(f"{result['label']}: per-point reflectance")
    ax[0].legend(fontsize=6, ncol=3)
    ax[1].set_title("transect spread")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig
