"""Scanning a folder of ASD files into blocks of replicate measurements."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .io import is_asd_path, read_asd

#: ``swir_vis_ratio`` above the threshold is a reference measurement, below it a
#: target. Raw DN and radiometrically calibrated files sit on different scales,
#: because calibration divides out the instrument responsivity, which is far
#: lower in the SWIR. Raw references land near 0.25 to 0.33 and raw targets from
#: 0.002 to 0.11; calibrated irradiance lands near 0.07, a calibrated target near
#: 0.002. Hence separate thresholds.
ROLE_THRESHOLD = {"RAW": 0.15}
ROLE_THRESHOLD_DEFAULT = 0.03
MIN_ROLE_RUN = 4
BLOCK_GAP_SECONDS = 90

FILE_RE = re.compile(r"^(?P<date>\d+)(?P<kind>[A-Za-z]+)(?P<setnum>\d*)$")

VIS_BAND = (450, 650)
SWIR_BAND = (1500, 1700)


def _smooth_roles(roles, min_run=MIN_ROLE_RUN):
    """Absorb short interior runs of the minority role into their neighbours.

    A handful of files classified against a long run either side is nearly always
    a transitional scan, not a real change of role.
    """
    roles = list(roles)
    if len(roles) < 2 * min_run:
        return roles
    runs, i = [], 0
    while i < len(roles):
        j = i
        while j + 1 < len(roles) and roles[j + 1] == roles[i]:
            j += 1
        runs.append([i, j, roles[i]])
        i = j + 1
    for k, (a, b, _) in enumerate(runs):
        if (b - a + 1) < min_run and 0 < k < len(runs) - 1 and runs[k - 1][2] == runs[k + 1][2]:
            for t in range(a, b + 1):
                roles[t] = runs[k - 1][2]
    return roles


@dataclass
class Dataset:
    """Every ASD file under a folder, with metadata and spectra."""

    inventory: pd.DataFrame
    spectra: dict
    wavelength: np.ndarray
    root: Path = field(default=None)

    def __len__(self):
        return len(self.inventory)

    def stack(self, block):
        """Replicates of one block as an (n, channels) array."""
        return np.array([self.spectra[f] for f in block["files"]])

    def mean(self, block, exclude=()):
        """Mean spectrum of a block, dropping excluded file indices.

        Returns ``(mean, kept_files)``.
        """
        drop = set(exclude)
        keep = [f for f, i in zip(block["files"], block["indices"]) if i not in drop]
        if not keep:
            raise ValueError(f"block {block['block_id']}: every spectrum excluded")
        return np.array([self.spectra[f] for f in keep]).mean(axis=0), keep


def scan_folder(folder, thresholds=None, default_threshold=ROLE_THRESHOLD_DEFAULT,
                min_role_run=MIN_ROLE_RUN, verbose=True) -> Dataset:
    """Read every ASD file under ``folder`` and classify each as reference or target."""
    thresholds = ROLE_THRESHOLD if thresholds is None else thresholds
    folder = Path(folder)
    records, spectra, wl_ref = [], {}, None

    for path in sorted(folder.rglob("*")):
        if not is_asd_path(path):
            continue
        try:
            m = read_asd(path)
        except Exception as exc:  # a stray file should not stop the scan
            warnings.warn(f"skipped {path.name}: {exc}")
            continue
        wl, s = m["wavelength"], m["spectrum"]
        if wl_ref is None:
            wl_ref = wl
        vis = np.nanmean(s[(wl >= VIS_BAND[0]) & (wl <= VIS_BAND[1])])
        swir = np.nanmean(s[(wl >= SWIR_BAND[0]) & (wl <= SWIR_BAND[1])])
        ratio = swir / vis if vis else np.nan
        parsed = FILE_RE.match(m["stem"])
        records.append(dict(
            file=m["file"], stem=m["stem"],
            kind=parsed["kind"] if parsed else "?",
            setnum=parsed["setnum"] if parsed else "",
            index=m["index"], time=m["datetime"], type=m["type"],
            it_ms=m["it"], swir1_gain=m["swir1_gain"], swir2_gain=m["swir2_gain"],
            swir1_offset=m["swir1_offset"], swir2_offset=m["swir2_offset"],
            splice1=m["splice1"], splice2=m["splice2"],
            swir_vis_ratio=ratio,
            role_raw="reference" if ratio > thresholds.get(m["type"], default_threshold)
            else "target",
            path=str(path),
        ))
        spectra[m["file"]] = s

    if not records:
        raise FileNotFoundError(f"no ASD files found under {folder}")

    df = pd.DataFrame(records).sort_values(["stem", "index"]).reset_index(drop=True)
    df["role"] = ""
    for _, group in df.groupby("stem"):
        df.loc[group.index, "role"] = _smooth_roles(
            group.sort_values("index")["role_raw"], min_role_run
        )
    changed = int((df["role"] != df["role_raw"]).sum())
    if changed and verbose:
        print(f"{changed} file(s) reassigned by run-length smoothing:")
        print(df.loc[df["role"] != df["role_raw"],
                     ["file", "swir_vis_ratio", "role_raw", "role"]].to_string(index=False))
    return Dataset(inventory=df, spectra=spectra, wavelength=wl_ref, root=folder)


def _make_block(block_id, stem, rows):
    first = rows[0]
    return dict(
        block_id=block_id, stem=stem, kind=first["kind"], setnum=first["setnum"],
        role=first["role"], it_ms=int(first["it_ms"]), type=first["type"],
        swir1_gain=int(first["swir1_gain"]), swir1_offset=int(first["swir1_offset"]),
        swir2_gain=int(first["swir2_gain"]), swir2_offset=int(first["swir2_offset"]),
        splice1=float(first["splice1"]), splice2=float(first["splice2"]),
        files=[r["file"] for r in rows], indices=[int(r["index"]) for r in rows],
        t_start=first["time"], t_end=rows[-1]["time"], n=len(rows),
    )


def find_blocks(dataset, gap_seconds=BLOCK_GAP_SECONDS):
    """Group consecutive files that share role, integration time, gain, offset and type.

    A block is the unit you average over. Splitting this way rather than assuming
    a fixed count per set matters: one file stem often holds several
    optimizations, and a reflectance day is usually opening panel, transect,
    closing panel under a single stem.
    """
    blocks, block_id = {}, 0
    for stem, group in dataset.inventory.groupby("stem", sort=True):
        group = group.sort_values("index")
        key, previous, rows = None, None, []
        for _, row in group.iterrows():
            this_key = (row["role"], row["it_ms"], row["swir1_gain"], row["swir1_offset"],
                        row["swir2_gain"], row["swir2_offset"], row["type"])
            gap = 0
            if previous is not None and row["time"] and previous["time"]:
                gap = (row["time"] - previous["time"]).total_seconds()
            if key is not None and (this_key != key or gap > gap_seconds):
                blocks[block_id] = _make_block(block_id, stem, rows)
                block_id += 1
                rows = []
            key, previous = this_key, row
            rows.append(row)
        if rows:
            blocks[block_id] = _make_block(block_id, stem, rows)
            block_id += 1
    return blocks


def block_table(blocks) -> pd.DataFrame:
    """One row per block, for review against field notes."""
    return pd.DataFrame([
        dict(block=b["block_id"], stem=b["stem"], role=b["role"], n=b["n"], type=b["type"],
             it_ms=b["it_ms"], swir1_gain=b["swir1_gain"], swir1_offset=b["swir1_offset"],
             idx=f"{b['indices'][0]}-{b['indices'][-1]}",
             start=b["t_start"].strftime("%Y-%m-%d %H:%M:%S") if b["t_start"] else "",
             end=b["t_end"].strftime("%H:%M:%S") if b["t_end"] else "")
        for b in blocks.values()
    ])


def detect_reflectance_stems(blocks):
    """Stems that look like a panel-plus-transect day rather than an albedo pair.

    A transect block is larger than its reference blocks; an albedo pair is
    roughly equal-sized.
    """
    out = []
    for stem in sorted({b["stem"] for b in blocks.values()}):
        refs = [b for b in blocks.values() if b["stem"] == stem and b["role"] == "reference"]
        tgts = [b for b in blocks.values() if b["stem"] == stem and b["role"] == "target"]
        if refs and tgts and max(b["n"] for b in tgts) > max(b["n"] for b in refs):
            out.append(stem)
    return out


def auto_pairs(blocks, skip_stems=()):
    """Pair each target block with the nearest reference block in the same stem.

    Returns ``(pairs, orphans)`` where pairs are ``(reference_id, target_id)``.
    """
    pairs, orphans = [], []
    for stem in sorted({b["stem"] for b in blocks.values()}):
        if stem in skip_stems:
            continue
        refs = [b for b in blocks.values() if b["stem"] == stem and b["role"] == "reference"]
        tgts = [b for b in blocks.values() if b["stem"] == stem and b["role"] == "target"]
        if refs and tgts:
            for t in tgts:
                r = min(refs, key=lambda x: abs((x["t_start"] - t["t_start"]).total_seconds())
                        if x["t_start"] and t["t_start"] else 0)
                pairs.append((r["block_id"], t["block_id"]))
        else:
            orphans += [b["block_id"] for b in refs + tgts]
    return pairs, orphans
