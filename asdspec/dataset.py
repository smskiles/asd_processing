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

#: Replicate sets are collected in runs, so a block of one or two files almost
#: always means the grouping went wrong rather than that the measurement did.
MIN_EXPECTED_BLOCK = 3

#: A real reference and target separate by more than an order of magnitude in the
#: SWIR-to-visible ratio. Below this factor the two groups are not distinct, so
#: the split is an artefact of the threshold rather than a property of the data.
ROLE_SEPARATION_MIN = 5.0

FILE_RE = re.compile(r"^(?P<date>\d+)(?P<kind>[A-Za-z]+)(?P<setnum>\d*)$")
OVERRIDE_RE = re.compile(r"^(?:(?P<folder>.+)/)?(?P<stem>[^/:]+)(?::(?P<lo>\d+)-(?P<hi>\d+))?$")

VIS_BAND = (450, 650)
SWIR_BAND = (1500, 1700)


def _runs(roles):
    """Contiguous runs of equal role, as ``[start, end, role]``."""
    runs, i = [], 0
    while i < len(roles):
        j = i
        while j + 1 < len(roles) and roles[j + 1] == roles[i]:
            j += 1
        runs.append([i, j, roles[i]])
        i = j + 1
    return runs


def _log_centroids(ratios, roles):
    """Median log ratio of each role group present."""
    roles = np.asarray(roles)
    out = {}
    for role in set(roles.tolist()):
        v = ratios[(roles == role) & np.isfinite(ratios) & (ratios > 0)]
        if v.size:
            out[role] = float(np.median(np.log(v)))
    return out


def _smooth_roles(roles, ratios, min_run=MIN_ROLE_RUN):
    """Reassign short runs that sit closer to the other role's ratio group.

    A short run is corrected only when its own ratio says it was classified
    wrongly, that is when it lies nearer the other group's centroid than its own.
    Under broken cloud the uplooking SWIR-to-visible ratio drops and can fall the
    wrong side of a fixed threshold, leaving a scatter of one-file runs; each of
    those sits far nearer the reference group than the target group, so each is
    put back.

    A short run whose ratio genuinely belongs where it was put is left alone, even
    at the edge of a set. A stray dark scan before a panel set is a real anomaly,
    and folding it into the panel average would corrupt that average silently,
    so it is left isolated for ``check_blocks`` to surface.

    Runs and centroids are re-derived after each correction, since one change
    moves both.
    """
    roles = list(roles)
    ratios = np.asarray(ratios, dtype=float)
    if len(roles) < 2 * min_run:
        return roles
    while True:
        runs = _runs(roles)
        centroids = _log_centroids(ratios, roles)
        if len(runs) < 2 or len(centroids) < 2:
            return roles
        for a, b, role in runs:
            if (b - a + 1) >= min_run:
                continue
            other = next(r for r in centroids if r != role)
            window = ratios[a:b + 1]
            window = window[np.isfinite(window) & (window > 0)]
            if window.size == 0:
                continue
            here = float(np.median(np.log(window)))
            if abs(here - centroids[other]) < abs(here - centroids[role]):
                for t in range(a, b + 1):
                    roles[t] = other
                break
        else:
            return roles


def _role_separation(ratios, roles):
    """How far apart the two role groups sit in SWIR-to-visible ratio.

    Returns the ratio of the reference group's median to the target group's, or
    infinity when only one role is present. A genuine reference and target differ
    by a factor of thirty or more; a value near 1 means the threshold cut through
    one population rather than between two.
    """
    ratios = np.asarray(ratios, dtype=float)
    roles = np.asarray(roles)
    ref = ratios[(roles == "reference") & np.isfinite(ratios)]
    tgt = ratios[(roles == "target") & np.isfinite(ratios)]
    if ref.size == 0 or tgt.size == 0:
        return np.inf
    lo = np.median(tgt)
    return np.inf if lo <= 0 else float(np.median(ref) / lo)


@dataclass
class Dataset:
    """Every ASD file under a folder, with metadata and spectra."""

    inventory: pd.DataFrame
    spectra: dict
    wavelength: np.ndarray
    root: Path = field(default=None)

    def __len__(self):
        return len(self.inventory)

    def spectrum(self, block, position):
        """One replicate of a block, by position within the block."""
        return self.spectra[block["paths"][position]]

    def stack(self, block):
        """Replicates of one block as an (n, channels) array."""
        return np.array([self.spectra[p] for p in block["paths"]])

    def mean(self, block, exclude=()):
        """Mean spectrum of a block, dropping excluded file indices.

        Returns ``(mean, kept_files)``.
        """
        drop = set(exclude)
        keep = [(p, f) for p, f, i in zip(block["paths"], block["files"], block["indices"])
                if i not in drop]
        if not keep:
            raise ValueError(f"block {block['block_id']}: every spectrum excluded")
        return (np.array([self.spectra[p] for p, _ in keep]).mean(axis=0),
                [f for _, f in keep])


def apply_role_overrides(df, overrides, verbose=True):
    """Force the role of chosen files, overriding the automatic classification.

    Keys are ``stem``, ``stem:first-last`` or ``folder/stem:first-last``; values
    are ``"reference"`` or ``"target"``. Because role is one of the fields blocks
    are cut on, an override also splits or merges blocks, which is how you repair
    a set the ratio test could not separate.

    Returns the number of rows changed.
    """
    if not overrides:
        return 0
    changed = 0
    for handle, role in overrides.items():
        if role not in ("reference", "target"):
            raise ValueError(f"role override {handle!r}: expected 'reference' or 'target', "
                             f"not {role!r}")
        m = OVERRIDE_RE.match(str(handle))
        if not m:
            raise ValueError(f"cannot parse role override key {handle!r}. Use 'stem', "
                             f"'stem:first-last' or 'folder/stem:first-last'.")
        sel = df["stem"] == m["stem"]
        if m["folder"]:
            sel &= df["folder"] == m["folder"]
        if m["lo"] is not None:
            sel &= df["index"].between(int(m["lo"]), int(m["hi"]))
        if not sel.any():
            raise KeyError(f"role override {handle!r} matches no files. Check the stem and "
                           f"index range against the inventory.")
        changed += int((df.loc[sel, "role"] != role).sum())
        df.loc[sel, "role"] = role
        if verbose:
            print(f"role override: {handle} -> {role} ({int(sel.sum())} files)")
    return changed


def scan_folder(folder, thresholds=None, default_threshold=ROLE_THRESHOLD_DEFAULT,
                min_role_run=MIN_ROLE_RUN, role_overrides=None, verbose=True) -> Dataset:
    """Read every ASD file under ``folder`` and classify each as reference or target.

    ``role_overrides`` forces the role of chosen files after the automatic pass;
    see :func:`apply_role_overrides` for the key format.
    """
    thresholds = ROLE_THRESHOLD if thresholds is None else thresholds
    folder = Path(folder).expanduser()
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
            date=parsed["date"] if parsed else "",
            folder=str(path.parent.relative_to(folder)) if path.parent != folder else ".",
            path=str(path),
        ))
        spectra[str(path)] = s

    if not records:
        raise FileNotFoundError(f"no ASD files found under {folder}")

    df = (pd.DataFrame(records)
          .sort_values(["folder", "stem", "index"])
          .reset_index(drop=True))

    repeated = df.groupby("stem")["folder"].nunique()
    repeated = repeated[repeated > 1]
    if len(repeated) and verbose:
        print(f"note: {len(repeated)} file stem(s) appear in more than one folder and are kept "
              f"as separate measurement sets: {', '.join(repeated.index)}")

    df["role"] = ""
    unstable = []
    for (folder_name, stem), group in df.groupby(["folder", "stem"]):
        ordered = group.sort_values("index")
        roles = _smooth_roles(ordered["role_raw"], ordered["swir_vis_ratio"].values,
                              min_role_run)

        # Collapse to one role only when the two groups do not actually separate.
        # A mixed set whose ratios differ by orders of magnitude is a real
        # reference/target split and must be kept, however ragged the runs were
        # before smoothing.
        ratios = ordered["swir_vis_ratio"]
        separation = _role_separation(ratios, roles)
        if separation < ROLE_SEPARATION_MIN:
            majority = max(set(roles), key=roles.count)
            unstable.append(dict(folder=folder_name, stem=stem, n=len(roles),
                                 ratio_min=float(np.nanmin(ratios)),
                                 ratio_max=float(np.nanmax(ratios)),
                                 separation=round(separation, 2),
                                 assigned=majority))
            roles = [majority] * len(roles)
        df.loc[ordered.index, "role"] = roles

    changed = int((df["role"] != df["role_raw"]).sum())
    if changed and verbose:
        print(f"{changed} file(s) reassigned during role classification")
    if unstable and verbose:
        print(f"\n{len(unstable)} file stem(s) did not separate into reference and target: their "
              f"SWIR-to-visible ratios differ by less than a factor of {ROLE_SEPARATION_MIN:g} "
              f"between the two groups, where a real pair differs by thirty or more. Each was "
              f"assigned a single role. Use ROLE_OVERRIDES if the split is real:")
        print(pd.DataFrame(unstable).to_string(index=False))

    apply_role_overrides(df, role_overrides, verbose=verbose)

    return Dataset(inventory=df, spectra=spectra, wavelength=wl_ref, root=folder)


def _make_block(block_id, stem, rows):
    first = rows[0]
    folder = first["folder"]
    prefix = "" if folder == "." else f"{folder}/"
    return dict(
        block_id=block_id, stem=stem, folder=folder, date=first["date"],
        kind=first["kind"], setnum=first["setnum"],
        role=first["role"], it_ms=int(first["it_ms"]), type=first["type"],
        swir1_gain=int(first["swir1_gain"]), swir1_offset=int(first["swir1_offset"]),
        swir2_gain=int(first["swir2_gain"]), swir2_offset=int(first["swir2_offset"]),
        splice1=float(first["splice1"]), splice2=float(first["splice2"]),
        files=[r["file"] for r in rows], paths=[r["path"] for r in rows],
        indices=[int(r["index"]) for r in rows],
        key=f"{prefix}{stem}:{int(rows[0]['index'])}-{int(rows[-1]['index'])}",
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
    for (_, stem), group in dataset.inventory.groupby(["folder", "stem"], sort=True):
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


def resolve_exclusions(blocks, exclude):
    """Normalise an exclusion dict keyed by block id or by stable block key.

    Block ids are assigned across the whole scan, so they shift when another
    folder is added. The ``key`` field (``stem:first-last``) does not, which
    makes it the safer thing to write down when processing many days at once.
    """
    if not exclude:
        return {}
    by_key = {b["key"]: bid for bid, b in blocks.items()}
    out = {}
    for handle, indices in exclude.items():
        if handle in blocks:
            out[handle] = list(indices)
        elif handle in by_key:
            out[by_key[handle]] = list(indices)
        else:
            raise KeyError(f"exclusion {handle!r} matches no block id or block key")
    return out


def block_table(blocks) -> pd.DataFrame:
    """One row per block, for review against field notes."""
    return pd.DataFrame([
        dict(block=b["block_id"], key=b["key"], folder=b["folder"], stem=b["stem"],
             role=b["role"], n=b["n"], type=b["type"],
             it_ms=b["it_ms"], swir1_gain=b["swir1_gain"], swir1_offset=b["swir1_offset"],
             idx=f"{b['indices'][0]}-{b['indices'][-1]}",
             start=b["t_start"].strftime("%Y-%m-%d %H:%M:%S") if b["t_start"] else "",
             end=b["t_end"].strftime("%H:%M:%S") if b["t_end"] else "")
        for b in blocks.values()
    ])


def resolve_pairs(blocks, pairs):
    """Normalise a list of pairs given as block ids or stable block keys.

    Accepts ``(reference, target)`` where each side is a block id or a key of the
    form ``folder/stem:first-last`` as ``block_table`` prints it. Keys are the
    safer thing to write down, since block ids shift when folders are added.
    """
    if not pairs:
        return []
    by_key = {b["key"]: bid for bid, b in blocks.items()}

    def resolve(handle, side):
        if handle in blocks:
            return handle
        if handle in by_key:
            return by_key[handle]

        # A key naming part of an existing block is the commonest mistake: the
        # user wants that block split, which pairing cannot do.
        m = OVERRIDE_RE.match(str(handle))
        if m and m["lo"] is not None:
            lo, hi = int(m["lo"]), int(m["hi"])
            for b in blocks.values():
                if b["stem"] != m["stem"]:
                    continue
                if m["folder"] and b["folder"] != m["folder"]:
                    continue
                if b["indices"][0] <= lo and hi <= b["indices"][-1]:
                    raise KeyError(
                        f"{side} {handle!r} names part of block {b['key']!r}, which is a "
                        f"single block of {b['n']} files. Pairing joins whole blocks and "
                        f"cannot split one. Split it first with ROLE_OVERRIDES, for example "
                        f"{{{handle!r}: 'reference'}}, and the two halves will pair on their "
                        f"own."
                    )
        near = [k for k in by_key if m and k.split("/")[-1].split(":")[0] == m["stem"]]
        hint = (f" Blocks for that stem: {near}." if near
                else f" Copy a key from the block table, such as "
                     f"{next(iter(by_key), 'stem:0-9')!r}.")
        raise KeyError(f"{side} {handle!r} matches no block id or block key.{hint}")

    out = []
    for ref, tgt in pairs:
        r, t = resolve(ref, "reference"), resolve(tgt, "target")
        if blocks[r]["role"] != "reference":
            raise ValueError(f"block {blocks[r]['key']} is a {blocks[r]['role']}, not a "
                             f"reference. Use ROLE_OVERRIDES to relabel it first.")
        if blocks[t]["role"] != "target":
            raise ValueError(f"block {blocks[t]['key']} is a {blocks[t]['role']}, not a "
                             f"target. Use ROLE_OVERRIDES to relabel it first.")
        if blocks[r]["type"] != blocks[t]["type"]:
            raise ValueError(f"block {blocks[r]['key']} is {blocks[r]['type']} but "
                             f"{blocks[t]['key']} is {blocks[t]['type']}. Their ratio is not "
                             f"an albedo.")
        out.append((r, t))
    return out


def merge_pairs(auto, manual=(), drop=()):
    """Combine automatic pairs with manual additions and removals.

    ``manual`` and ``drop`` must already be resolved to block ids by
    :func:`resolve_pairs`. A manual pair replaces any automatic pair that used
    the same target, since a target belongs to one reference.
    """
    manual = list(manual)
    dropped = {tuple(p) for p in drop}
    manual_targets = {t for _, t in manual}
    kept = [p for p in auto
            if tuple(p) not in dropped and p[1] not in manual_targets]
    replaced = len(auto) - len(kept) - sum(1 for p in auto if tuple(p) in dropped)
    if replaced:
        print(f"{replaced} automatic pair(s) replaced by a manual pair on the same target")
    return kept + manual


def check_blocks(blocks, min_expected=MIN_EXPECTED_BLOCK, verbose=True):
    """Report blocks too small to be a real replicate set.

    Replicates are collected in runs, so a block of one or two files means the
    grouping went wrong. The usual cause is the same file stem appearing in two
    folders, which interleaves on file index; the folder column in
    ``block_table`` shows whether that is what happened.
    """
    small = [b for b in blocks.values() if b["n"] < min_expected]
    if small and verbose:
        print(f"WARNING: {len(small)} of {len(blocks)} blocks have fewer than {min_expected} "
              f"files. Replicate sets are collected in runs, so this usually means the grouping "
              f"is wrong rather than the measurement. Check the folder and stem columns:")
        print(block_table({b["block_id"]: b for b in small})
              [["block", "key", "folder", "stem", "role", "n", "it_ms", "start"]]
              .to_string(index=False))
    return small


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


PAIR_WARN_MINUTES = 15


def resolve_reflectance_stems(blocks, force_reflectance=(), force_albedo=(), verbose=True):
    """Decide which stems are reflectance days, with manual overrides.

    Auto-detection treats a stem as a reflectance day when its largest target
    block is bigger than its largest reference block, which is what separates a
    panel-plus-transect day from an albedo pair. That test misses a transect no
    longer than its panel sets, and misfires on an albedo day with an unusually
    long downlooking set.

    ``force_reflectance`` adds stems to the detected set; ``force_albedo``
    removes them. Both are checked against the blocks, so a typo raises rather
    than being ignored.
    """
    known = {b["stem"] for b in blocks.values()}
    for name in list(force_reflectance) + list(force_albedo):
        if name not in known:
            raise KeyError(f"stem {name!r} is not in this dataset. Stems present: "
                           f"{sorted(known)}")

    detected = set(detect_reflectance_stems(blocks))
    stems = (detected | set(force_reflectance)) - set(force_albedo)

    for stem in sorted(set(force_reflectance) - detected):
        roles = {b["role"] for b in blocks.values() if b["stem"] == stem}
        if roles != {"reference", "target"}:
            raise ValueError(
                f"stem {stem!r} was forced to a reflectance day but has only {sorted(roles)} "
                f"blocks. A reflectance day needs a panel (reference) and a transect (target); "
                f"fix the roles with ROLE_OVERRIDES first."
            )
    if verbose:
        added = sorted(set(force_reflectance) - detected)
        removed = sorted(detected & set(force_albedo))
        if added:
            print(f"forced to reflectance days: {added}")
        if removed:
            print(f"forced to albedo days: {removed}")
    return sorted(stems)


def auto_pairs(blocks, skip_stems=(), group_by="date", warn_minutes=PAIR_WARN_MINUTES,
               verbose=True):
    """Pair each target block with the nearest reference block in time.

    ``group_by="date"`` matches within one acquisition day and folder, which is
    what you want when uplooking and downlooking sets carry different file stems,
    for example ``240411Au`` and ``240411Ad``. ``group_by="stem"`` restricts
    pairing to blocks sharing a file stem, which suits naming schemes that keep
    both halves of a pair under one stem.

    A reference is only paired with a target of the same spectrum type. Raw DN
    over calibrated irradiance is not a ratio of anything, and a calibrated
    irradiance set often sits closer in time to a raw target than the raw
    reference does, so nearest-in-time alone picks the wrong one.

    Returns ``(pairs, orphans)`` where pairs are ``(reference_id, target_id)``.
    Pairs separated by more than ``warn_minutes`` are reported: illumination
    drifts, so a reference taken long before or after its target is worth a look.
    """
    if group_by not in ("date", "stem"):
        raise ValueError(f"group_by must be 'date' or 'stem', not {group_by!r}")

    def group_of(b):
        return (b["folder"], b["date"] or b["stem"]) if group_by == "date" else (b["folder"], b["stem"])

    usable = [b for b in blocks.values() if b["stem"] not in skip_stems]
    pairs, orphans, far = [], [], []
    for key in sorted({group_of(b) for b in usable}):
        members = [b for b in usable if group_of(b) == key]
        refs = [b for b in members if b["role"] == "reference"]
        tgts = [b for b in members if b["role"] == "target"]
        if not (refs and tgts):
            orphans += [b["block_id"] for b in members]
            continue
        used, unmatched = set(), []
        for t in tgts:
            candidates = [b for b in refs if b["type"] == t["type"]]
            if not candidates:
                unmatched.append(t["block_id"])
                continue
            r = min(candidates, key=lambda x: abs((x["t_start"] - t["t_start"]).total_seconds())
                    if x["t_start"] and t["t_start"] else 0)
            pairs.append((r["block_id"], t["block_id"]))
            used.add(r["block_id"])
            if r["t_start"] and t["t_start"]:
                gap = abs((r["t_start"] - t["t_start"]).total_seconds()) / 60
                if gap > warn_minutes:
                    far.append((r["block_id"], t["block_id"], gap))
        orphans += unmatched
        orphans += [b["block_id"] for b in refs if b["block_id"] not in used]

    if far and verbose:
        print(f"note: {len(far)} pair(s) have reference and target more than {warn_minutes:g} "
              f"minutes apart. Illumination drifts, so check these against field notes:")
        for r, t, gap in far:
            print(f"   blocks {r} and {t}: {gap:.0f} min apart")
    return pairs, sorted(set(orphans))
