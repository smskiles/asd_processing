# asdspec

Processing of ASD FieldSpec binary spectra into spectral albedo, broadband albedo, and
panel-referenced reflectance.

Reads `.asd` binaries directly, so no RS3 or ViewSpec export step is needed. Handles the parts of
the workflow that are easy to get wrong: grouping replicates into the right measurement blocks,
flagging outliers, correcting for integration time and detector splices, and weighting the
broadband integral with a proper irradiance spectrum.

## Install

```bash
git clone https://github.com/smskiles/asd_processing.git
cd asd_processing
pip install -r requirements.txt
```

In Colab:

```python
!pip install -q git+https://github.com/smskiles/asd_processing.git
```

## Quick start

```python
import asdspec as asd

data = asd.scan_folder("path/to/spectra")      # reads every .NNN file, recursively
blocks = asd.find_blocks(data)                 # groups replicates into measurement blocks
print(asd.block_table(blocks))                 # check this against your field notes

pairs, orphans = asd.auto_pairs(blocks, asd.detect_reflectance_stems(blocks))
results = [asd.compute_albedo(data, blocks[r], blocks[t]) for r, t in pairs]

print(asd.splice_qc_table(results))            # per-set QC
```

## The notebook

`notebooks/asd_processing.ipynb` drives the whole workflow, including QC plots, broadband
integration and export. Open it in Colab with:

```
https://colab.research.google.com/github/smskiles/asd_processing/blob/main/notebooks/asd_processing.ipynb
```

**The notebook and the package are a matched pair.** The first cell checks `asdspec.__version__`
against the version the notebook was written for and stops with an explanation if they differ, rather
than letting a call fail later with a confusing `TypeError`. After changing the package, push it
before re-running the notebook.

It opens read-only, so save a copy to Drive before editing. The first cell detects Colab and
installs the package and mounts Drive on its own; from a local clone it adds the repository to the
path instead, so no edit is needed either way. Set `DATA_DIR` and `OUTPUT_DIR` in the settings cell
and run the notebook through.

## What the code does

### Blocks

A block is a run of consecutive files sharing role, integration time, SWIR gain and offset, and
spectrum type, with no long pause. That is the unit averaged over.

Grouping this way rather than assuming a fixed count per set matters in practice. One file stem
often contains several optimizations, and a reflectance day is typically opening panel, transect,
closing panel all under one stem.

`check_blocks` reports any block with fewer than three files. Replicates are collected in runs, so a
block of one or two means the grouping went wrong rather than the measurement. The usual cause is a
stem appearing in two folders.

Each file is classified as **reference** (the bright, solar-shaped denominator: an uplooking cosine
receptor, or a Spectralon panel) or **target**, using the ratio of mean 1500 to 1700 nm to mean 450
to 650 nm. Snow and glacier ice are nearly black in the SWIR, so targets land from about 0.002 to
0.11 while references sit near 0.25 to 0.33. Raw DN and radiometrically calibrated files use
separate thresholds, since calibration divides out the instrument responsivity and moves the whole
scale. A short run is reassigned only when its own ratio says it was classified wrongly, that is when it
sits nearer the other group's centroid than its own. This matters under broken cloud: the uplooking
ratio drops with cloud and individual scans fall the wrong side of a fixed threshold, leaving a
scatter of one-file runs. Each of those sits far nearer the reference group than the target group, so
each is put back, and the uplooking set stays whole instead of being swallowed by the adjacent
downlooking block.

A short run whose ratio genuinely belongs where it was put is left alone, even at the edge of a set.
A stray dark scan before a panel set is a real anomaly, and folding it into the panel average would
corrupt that average silently, so it is left as its own block for `check_blocks` to surface.

A stem is collapsed to a single role only when its two groups do not actually separate, measured as
the ratio between their median SWIR-to-visible values. A real reference and target differ by a factor
of thirty or more, so anything under five means the threshold cut through one population rather than
between two. Such stems are reported, with their separation, so they can be checked or fixed with
`role_overrides`. A mixed set whose ratios differ by orders of magnitude is always kept split,
however ragged its runs look before smoothing.

### Scanning several dates at once

`scan_folder` searches recursively, so pointing it at a season folder picks up every dated
subfolder in one pass. Blocks are grouped by file stem, and ASD stems normally carry the date
(`240411a`, `240424a`), so days stay separate and `auto_pairs` pairs within each day.

Blocks are grouped by folder *and* stem, so the same stem appearing in two folders, for instance a
morning and an afternoon session archived separately, stays two measurement sets rather than being
interleaved into one. `scan_folder` prints a note when it sees a stem in more than one folder.

Block ids are assigned across the whole scan, so they shift when a folder is added or removed. Every
block also carries a stable `key` of the form `folder/stem:first-last`, shown in `block_table`. Key
exclusions by that rather than by id when processing many days, and pass them through
`resolve_exclusions`, which accepts either.

### Pairing

`auto_pairs` matches each target block with the nearest reference block in time, within one
acquisition day and folder. Grouping by day rather than by stem means naming schemes that split the
two halves of a pair across stems work, for example `240411Au` uplooking and `240411Ad` downlooking.
Pass `group_by="stem"` to restrict pairing to blocks sharing a stem.

A reference is only paired with a target of the same spectrum type. Raw DN over calibrated
irradiance is not a ratio of anything, and a calibrated irradiance set often sits closer in time to a
raw target than the raw reference does, so nearest-in-time alone picks the wrong one. A calibrated
irradiance reference does pair with a calibrated reflected set, which is a valid albedo.

Pairs whose reference and target are more than 15 minutes apart are reported, since illumination
drifts. Unpaired blocks are listed rather than silently dropped.

### Fixing what the automatic rules get wrong

Three overrides, all keyed by the stable block key (`folder/stem:first-last`) that `block_table`
prints, or by block id. Apply them in this order, since each changes what the next one sees.

**Role**, when a block is labelled wrong, or when one stem was cut into ragged pieces because its
ratio straddles the threshold. Role is one of the fields blocks are cut on, so an override also
re-cuts the blocks:

```python
data = asd.scan_folder(path, role_overrides={"240401A:0-9": "reference",
                                             "240401A:10-19": "target"})
```

A bare stem sets the whole stem (`{"240509i": "reference"}`); a folder prefix disambiguates a stem
that appears in two folders (`{"morning/240411Au:0-9": "reference"}`).

**Day type**, when a day is processed as the wrong kind. A reflectance day divides one averaged panel
into every transect point; an albedo day averages the target set and divides once. Auto-detection
calls a stem a reflectance day when its largest target block is bigger than its largest reference
block, which misses a transect no longer than its panel sets and misfires on an albedo day with an
unusually long downlooking set:

```python
stems = asd.resolve_reflectance_stems(blocks,
                                      force_reflectance=["240509A3"],
                                      force_albedo=["240522a6"])
```

Both take bare stems and are applied after auto-detection rather than replacing it, so forcing one
day does not un-detect the others. A stem that is not in the data, or one forced to a reflectance day
without both a panel and a transect, raises rather than being ignored.

**Pairing**, when two blocks are correctly labelled but were not paired, for instance a reference
and target under different stems:

```python
pairs = asd.merge_pairs(
    auto,
    manual=asd.resolve_pairs(blocks, [("240509A2:0-8", "240509A3:0-19")]),
    drop=asd.resolve_pairs(blocks, [("240522a5:0-9", "240522a5:20-29")]),
)
```

A manual pair replaces any automatic pair on the same target, since a target belongs to one
reference. `resolve_pairs` refuses a pair whose blocks have the wrong roles or mismatched spectrum
types, and a key matching nothing raises an error naming the problem rather than failing quietly
later.

### QC

Replicates are scored two ways, both as modified z-scores built on the median absolute deviation
rather than the standard deviation: with ten replicates, one bad spectrum inflates a standard
deviation enough to hide itself.

- **level**: mean over 450 to 1300 nm. Catches a scan taken while a cloud crossed, or a saturated one.
- **shape**: each spectrum divided by its own mean, compared to the block median. Catches a shadow
  on the receptor, a tipped sensor, or an obstruction in the field of view.

A steady drift in level across a reference set is normal as the sun moves. Nothing is dropped
automatically.

### Integration time

ASD raw DN in the VNIR (350 to 1000 nm) scales linearly with integration time. The SWIR uses gain
and offset instead and does not respond to it. So when a reference and target were collected at
different integration times, the VNIR of the ratio is wrong by exactly that factor and the SWIR is
not. The correction is applied below the first splice only, and only to raw DN files.

The linearity is worth checking on your own instrument rather than assuming: point the foreoptic at
a stable target under steady illumination and take a set at each integration time you use, then
compare the VNIR ratio to the nominal one using the SWIR as a tie-point. On a FieldSpec 4 this
returns the nominal ratio to within a few percent.

**The better fix is upstream: do not re-optimize between the two measurements you are going to
ratio.** When integration time is matched the factor is exactly 1 and the problem disappears.

### Splice correction

The three detectors do not agree perfectly at their boundaries, so a ratio spectrum carries a small
step at 1000 nm and a larger one at 1800 nm. On a FieldSpec 4 the 1000 nm step runs about 1.5 to
2 percent downward from VNIR to SWIR1 in matched-integration-time data. That is a few channels'
worth of real spectral slope compressed into one channel, which reads as a visible notch in a stack
of spectra, particularly where albedo is low.

### Two ways to close the step, and why the default changed

The step comes from a small additive offset, around 20 DN, in the VNIR: residual dark current or
stray light. That matters because the reflectance error such an offset causes depends on how much
signal there is, and the signal collapses toward 1000 nm. On a FieldSpec 4 a Spectralon panel reads
about 41000 DN at 500 nm and about 1200 DN at 1000 nm. The same 20 DN is therefore negligible in the
visible and several percent at the splice. Measured on a dark glacier-ice transect:

| wavelength | panel DN | error from a 19 DN offset |
|---|---|---|
| 400 nm | 16272 | +0.17 % |
| 500 nm | 41477 | +0.06 % |
| 850 nm | 12230 | +0.34 % |
| 950 nm | 1825 | +3.50 % |
| 1000 nm | 1214 | +8.25 % |

**`SPLICE_MODE = "offset"`, the default, subtracts that offset in DN from both spectra before they
are ratioed.** It removes the error where it exists and leaves the rest of the VNIR untouched.

**`"swir1_anchor"` scales the whole VNIR by one factor instead**, which is what ASD's own software
does. It closes the step equally well, but it pays for a boundary-sized correction with a uniform
change across the entire VNIR. On that same ice transect it drags 500 nm reflectance from 0.214 to
0.196, an 8 percent error in the part of the spectrum where light-absorbing particles are measured.
Over bright snow the same mode costs about 1.4 percent.

This was the default until version 1.4.0, so numbers produced before then are low in the visible,
most of all for dark targets. Use `"swir1_anchor"` deliberately when the point is to match an RS3 or
ViewSpec export: those apply the multiplicative join, and the offset mode disagrees with them by
about 8 percent in the visible on a dark target for exactly this reason.

Both modes fit each side of the boundary locally and extrapolate to it, so the real spectral slope
stays out of the estimate. Comparing window means either side instead would bake the slope in: over
snow near 1000 nm that alone accounts for roughly three quarters of the apparent step.

The subtraction is refused, and the spectrum left alone, when the implied offset is negative or
larger than half the smallest VNIR signal, since neither can be an instrument offset. It warns when
the offset is far from 20 DN, which usually means an integration time that was not accounted for
rather than an offset at all.

The 1800 nm boundary is left alone by default. Over snow it sits in a low-signal,
water-vapour-affected region where the factor is poorly constrained. Turn it on for bright targets.

**Why the step is bigger over dark targets.** Near 1000 nm the silicon detector response has
collapsed, so the raw DN there is small: on a FieldSpec 4, roughly 1200 DN for a Spectralon panel
against 14000 DN on the SWIR side of the boundary. Any additive offset, residual dark current or
stray light, is therefore a large fraction of the VNIR signal, and it inflates a ratio whose
numerator is much smaller than its denominator. The darker the target, the bigger the inflation and
the bigger the step.

Solving the step for that offset turns it into a number that should be roughly constant for an
instrument, whatever the target. Measured three ways on one FieldSpec 4, it is consistent:

| measurement | reflectance near 1000 nm | splice step | implied offset |
|---|---|---|---|
| snow, cosine receptor | 0.43 | -1.4 % | 22 DN |
| snow, cosine receptor | 0.42 | -1.3 % | 19 DN |
| glacier ice, panel referenced | 0.08 | -8.4 % | 19 DN |
| panel against panel | 1.0 | +0.25 % | n/a |

`implied_vnir_offset` reports it, and `splice_qc_table` carries it as `vnir_offset_dn`. A value near
20 means the step is ordinary and the correction is doing its job, however large the step looks. A
very different value means something else is wrong, most often an integration time that was not
accounted for. The panel-against-panel row is the useful control: two bright spectra show almost no
step, which is why a large step is a property of the target rather than of the panels.

**The VNIR splice factor is the most useful QC number here.** After a correct integration-time
correction it lands near 0.985. A value well outside 0.96 to 1.01 means the integration-time
correction was wrong for that pair, usually because the instrument was re-optimized between
reference and target. `splice_qc_table` reports it per set.

Note that the two corrections both act on the VNIR and cannot be fully separated after the fact.
When integration time was mismatched, the splice correction absorbs both the instrument step and any
residual integration-time error. The spectrum comes out continuous either way, but its absolute VNIR
level then rests on the assumption that the true spectrum is smooth across 1000 nm.

### Masking

Fixed windows (the blue end, the 1350 to 1450 and 1800 to 1950 nm water vapour bands, past 2350 nm)
plus a data-driven mask wherever replicate scatter is large relative to the signal. Over melting snow
the downlooking SWIR falls to a few DN and the ratio becomes unstable. Masked channels are
interpolated so the result can be integrated, and the mask is returned so plots can leave them blank.

Separately, treat 960 to 1010 nm as the weakest part of the spectrum whatever the splice correction
does, since the silicon detector response is falling off steeply at its long-wavelength end.

### Broadband albedo

The irradiance-weighted mean of the spectral albedo. The weighting spectrum must be in physical
units: **raw uplooking DN will not do**, because DN is irradiance times the instrument's spectral
responsivity times integration time, and that responsivity is strongly wavelength dependent, so
weighting by DN silently reweights the integral.

Two sources are supported. A measured irradiance set, collected with the irradiance calibration
loaded so the ASD writes it with the IRRADIANCE type flag. Or a modeled clear-sky table indexed by
solar zenith angle, for days when irradiance was not measured. For the modeled route, supply either
a zenith angle directly or the site latitude, longitude and UTC offset, in which case the angle is
computed per set from the file timestamps. ASD headers store local wall-clock time, which is why the
offset is needed.

Two things worth knowing:

- **The zenith angle barely matters.** Weights are normalised, so only the shape of the spectrum
  enters, and shape changes slowly with zenith angle. Across a 25 to 50 degree table the resulting
  broadband albedo typically varies by under 0.001. Nearest-column is sufficient.
- **Measured against modeled matters more.** Differences of a few hundredths are normal, because a
  clear-sky model cannot know the day's cloud, aerosol and diffuse fraction, and a bluer measured
  spectrum pushes snow albedo up. Do not mix modeled and measured results in one analysis without
  saying so.

`ModeledIrradiance.from_csv` expects a table whose first column is wavelength and whose remaining
columns are named with a zenith angle, for example `Z25`. Wavelength in micrometres or nanometres is
detected automatically, as is whether values are spectral irradiance per nanometre or integrated per
wavelength bin, since only one of those integrates to a plausible broadband total.

Coverage: the instrument sees 350 to 2500 nm, and roughly 3 to 5 percent of incoming solar energy
falls outside that window, mostly below 350 nm and above 2500 nm where snow albedo is low. A
broadband albedo over 350 to 2500 nm therefore slightly overestimates a true 300 to 4000 nm value.
Report the integration limits.

### Reflectance

The white reference set is averaged after outlier removal. Transect points are not averaged, since
the variability along the transect is the signal, so reflectance is computed per point against the
averaged panel. The splice factor is derived once from the block mean and applied to every point,
rather than solved per point, which would be unstable on the darkest targets.

`choose_panel` prefers a panel whose gain and offset match the transect, because a panel taken after
a re-optimization cannot be used for the SWIR. With an opening and closing panel, `panel_drift` gives
a free stability check on the whole transect.

Two caveats. The default panel reflectance of 0.99 flat is a placeholder: Spectralon is close to that
in the visible but falls off in the SWIR, and each panel has its own calibration certificate. And
with a narrow foreoptic under natural illumination this is an HDRF, not a bi-hemispherical albedo;
over snow and ice, where the forward scattering lobe is strong, the two are not interchangeable.

## File format note

An ASD file is a fixed 484-byte header followed by the spectrum. The GPS block in that header is
**128 bytes, not 32**. Several published parsers use 32, which shifts the integration time and SWIR
gain and offset fields 96 bytes early so they read back as zero. Those are exactly the fields needed
for a correct albedo, and the failure is silent. `tests/test_io.py` guards this.

## Tests

```bash
pip install pytest
pytest
```

The tests build synthetic ASD binaries, so they run without field data. They cover the header
layout, the block and role logic, the integration-time and splice corrections against known injected
errors, the solar position calculation against orbital geometry, and irradiance table parsing.

## Repository layout

```
asdspec/            package
  io.py             ASD binary reader
  dataset.py        folder scanning, role classification, block detection, pairing
  qc.py             outlier scoring
  corrections.py    integration time, splice, masking
  albedo.py         spectral and broadband albedo
  reflectance.py    panel-referenced transect reflectance
  irradiance.py     solar position, modeled clear-sky tables
  plots.py          figures
notebooks/          driver notebook
nbcells/            notebook cell sources, assembled by build_notebook.py
tests/              pytest suite, synthetic data only
```

The notebook is generated from `nbcells/` by `build_notebook.py`, which keeps diffs readable. Edit
the cell files and rebuild rather than committing notebook JSON changes by hand.

## License

Add a license before publishing.
