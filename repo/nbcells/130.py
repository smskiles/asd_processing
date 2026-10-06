model = None
model_path = MODELED_IRRADIANCE_FILE
if model_path is None:
    candidates = [p for p in list(DATA_DIR.rglob("*.csv")) + list(Path.cwd().glob("*.csv"))
                  if "irrad" in p.name.lower()]
    model_path = candidates[0] if candidates else None
if model_path is not None:
    model = asd.ModeledIrradiance.from_csv(model_path, WL)
    print(model.describe())

irradiance_blocks = [b for b in blocks.values()
                     if b["type"] == "IRRADIANCE" and b["role"] == "reference"]
E_measured = None
if irradiance_blocks:
    block = irradiance_blocks[0]
    raw, kept = data.mean(block, EXCLUDE.get(block["block_id"], ()))
    E_measured, _ = asd.mask_fill(WL, raw, windows=[(2400, 2500)])
    E_measured = np.clip(E_measured, 0, None)
    print(f"measured irradiance: block {block['block_id']} ({block['stem']}, n={len(kept)}, "
          f"{block['t_start']:%H:%M:%S}), {TRAPZ(E_measured, WL):.0f} W m-2 over 350-2500 nm")
else:
    print("no calibrated irradiance set on this day")


def zenith_for(when):
    if MANUAL_SZA is not None:
        return float(MANUAL_SZA)
    if when is None or None in (SITE_LAT, SITE_LON, UTC_OFFSET_HOURS):
        return None
    return asd.solar_zenith(when, SITE_LAT, SITE_LON, UTC_OFFSET_HOURS)


def weighting_for(result):
    if IRRADIANCE_SOURCE in ("auto", "measured") and E_measured is not None:
        return dict(E=E_measured, source="measured", sza=np.nan, column="", clamped=False)
    if IRRADIANCE_SOURCE == "measured" or model is None:
        return dict(E=None, source="unavailable", sza=np.nan, column="", clamped=False)
    sza = zenith_for(result["time"])
    E, angle, clamped = model.for_sza(sza)
    if E is None:
        return dict(E=None, source="unavailable", sza=np.nan, column="", clamped=False)
    return dict(E=E, source="modeled", sza=round(sza, 1), column=f"Z{angle:g}", clamped=clamped)


if E_measured is not None and model is not None and results:
    sza = zenith_for(results[0]["time"])
    E_model, angle, _ = model.for_sza(sza if sza is not None else np.median(model.angles))
    plots.plot_irradiance_comparison(WL, E_measured, E_model, f"modeled Z{angle:g}")
    plt.show()
    for lo, hi in [(350, 700), (700, 1000), (1000, 1500), (1500, 2500)]:
        m = (WL >= lo) & (WL <= hi)
        a = TRAPZ(E_measured[m], WL[m]) / TRAPZ(E_measured, WL)
        b = TRAPZ(E_model[m], WL[m]) / TRAPZ(E_model, WL)
        print(f"   {lo}-{hi} nm: measured {100 * a:5.1f} %, modeled {100 * b:5.1f} % "
              f"({100 * (b - a):+.1f} points)")
