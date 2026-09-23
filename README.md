# ASD Spectroscopy Processing for Snow and Ice - Albedo and Reflectance

Processing of ASD FieldSpec binary spectra (original file format) into spectral albedo, broadband albedo, and
panel-referenced reflectance.

Reads `.asd` binaries directly, so no RS3 or ViewSpec export step is needed. Handles the parts of
the workflow that are easy to get wrong: grouping replicates into the right measurement blocks,
flagging outliers, correcting for integration time and detector splices, and weighting the
broadband integral with a proper irradiance spectrum.

## Install

```bash
git clone https://github.com/<user>/asdspec.git
cd asdspec
pip install -r requirements.txt
```

In Colab:

```python
!pip install -q git+https://github.com/<user>/asdspec.git
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

`notebooks/asd_processing.ipynb` drives the whole workflow, including plots, broadband integration
and export. Edit the settings block at the top and run it through.

## What the code does

### Blocks

A block is a run of consecutive files sharing role, integration time, SWIR gain and offset, and
spectrum type, with no long pause. That is the unit averaged over.

Grouping this way rather than assuming a fixed count per set matters in practice. One file stem
can contain several optimizations, and a reflectance transect is typically opening panel, transect,
closing panel all under one stem.

Each file is classified as **reference** (the bright, solar-shaped denominator: an uplooking cosine
receptor, or a Spectralon panel) or **target**, using the ratio of mean 1500 to 1700 nm to mean 450
to 650 nm. Snow and glacier ice are nearly black in the SWIR, so targets land from about 0.002 to
0.11 while references sit near 0.25 to 0.33. Raw DN and radiometrically calibrated files use
separate thresholds, since calibration divides out the instrument responsivity and moves the whole
scale. Short runs of the minority role inside a long run are absorbed, because those are almost
always transitional scans rather than a real change of role.

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
different integration times, the VNIR of the ratio is wrong by that factor and the SWIR is
not. The correction is applied below the first splice only, and only to raw DN files.

The linearity is worth checking on your own instrument rather than assuming: point the foreoptic at
a stable target under steady illumination and take a set at each integration time you use, then
compare the VNIR ratio to the nominal one using the SWIR as a tie-point. On a FieldSpec 4 this
returns the nominal ratio to within a few percent.

**The better fix is to not re-optimize between the two measurements you are going to
ratio.**

### Splice correction

The three detectors do not agree perfectly at their boundaries, so a ratio spectrum carries a small
step at 1000 nm and a larger one at 1800 nm. On a FieldSpec 4 the 1000 nm step runs about 1.5 to
2 percent downward from VNIR to SWIR1 in matched-integration-time data. That is a few channels'
worth of real spectral slope compressed into one channel, which reads as a visible notch in a stack
of spectra, particularly where albedo is low.

`splice_correct` scales the VNIR to meet SWIR1 and leaves SWIR1 alone, which is what the instrument
software does. Both sides of the boundary are fitted locally and extrapolated to it, so the real
spectral slope stays out of the estimate. Comparing window means either side instead would bake the
slope in: over snow near 1000 nm that alone accounts for roughly three quarters of the apparent step.

The 1800 nm boundary is left alone by default. Over snow it sits in a low-signal,
water-vapour-affected region where the factor is poorly constrained. Turn it on for bright targets.

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

Separately, treat 960 to 1010 nm as a weaker part of the spectrum, since the silicon detector response is 
falling off steeply at its long-wavelength end.

### Broadband albedo

The irradiance-weighted mean of the spectral albedo. The weighting spectrum must be in physical
units: **do not use raw uplooking DN**, because DN is irradiance times the instrument's spectral
responsivity times integration time, and that responsivity is strongly wavelength dependent, so
weighting by DN silently reweights the integral.

Two sources are supported. A measured irradiance set, collected with the irradiance calibration
loaded so the ASD writes it with the IRRADIANCE type flag. Or a modeled clear-sky table indexed by
solar zenith angle, for days when irradiance was not measured (or measured w/ an uncalibrated ASD). 
For the modeled route, supply either a zenith angle directly or the site latitude, longitude and UTC 
offset, in which case the angle is computed per set from the file timestamps. ASD headers store local 
time, which is why the offset is needed.

Two things worth knowing:

- **The zenith angle has little practical matter for albedo, but would matter for net solar.**
  Weights are normalised, so only the shape of the spectrum is relevant, and shape changes slowly
  with zenith angle.
  Across a 25 to 50 degree table the resulting broadband albedo typically varies by under 0.001.
  Nearest-column is sufficient.
- **Use measured if you have it.** A clear-sky model cannot know the day's cloud, aerosol and diffuse
  fraction, and a 'bluer' measured spectrum will increase broadband albeod.
  Do not mix modeled and measured results in one analysis.

`ModeledIrradiance.from_csv` expects a table whose first column is wavelength and whose remaining
columns are named with a zenith angle, for example `Z25`. Wavelength in micrometres or nanometres is
detected automatically, as is whether values are spectral irradiance per nanometre or integrated per
wavelength bin, since only one of those integrates to a plausible broadband total.


### Reflectance

The white reference set is averaged after outlier removal. Transect points are not averaged, since
the variability along the transect is the signal, so reflectance is computed per point against the
averaged panel. The splice factor is derived once from the block mean and applied to every point,
rather than solved per point, which would be unstable on the darkest targets.

`choose_panel` prefers a panel whose gain and offset match the transect, because a panel taken after
a re-optimization cannot be used for the SWIR. With an opening and closing panel, `panel_drift` gives
a free stability check on the whole transect.

Two caveats: The default panel reflectance of 0.99 flat is a placeholder, Spectralon is close to that
in the visible but falls off in the SWIR, and each panel has its own calibration certificate. 
With a narrow foreoptic under natural illumination **this is an HDRF**, not a bi-hemispherical albedo;
over snow and ice, where the forward scattering is strong, the two are not interchangeable.

## File format note

An ASD file is a fixed 484-byte header followed by the spectrum. The GPS block in that header is
**128 bytes, not 32**. Several published parsers use 32, which shifts the integration time and SWIR
gain and offset fields 96 bytes early so they read back as zero. Those are the fields needed
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

MIT License

Copyright (c) 2026 S. McKenzie Skiles

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
