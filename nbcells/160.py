if PANEL_FILE:
    panel_curve = pd.read_csv(PANEL_FILE)
    panel_reflectance = np.interp(WL, panel_curve["wavelength_nm"], panel_curve["reflectance"])
else:
    panel_reflectance = PANEL_REFLECTANCE

reflectance_results = []
for stem in reflectance_stems:
    refs = [b for b in blocks.values() if b["stem"] == stem and b["role"] == "reference"]
    targets = sorted([b for b in blocks.values() if b["stem"] == stem and b["role"] == "target"],
                     key=lambda b: b["t_start"])
    print(f"\n{stem}: {len(refs)} panel block(s), {len(targets)} transect block(s)")

    drift = asd.panel_drift(data, blocks, stem)
    if drift is not None:
        ratio, same_settings, minutes = drift
        summary = "  ".join(f"{w} nm {ratio[int(w - WL[0])]:.3f}" for w in (500, 900, 1500, 2000))
        print(f"  panel drift over {minutes:.1f} min (closing / opening): {summary}")
        if not same_settings:
            print("  the panels were taken with different SWIR gain or offset, so the SWIR "
                  "figures above are a settings change rather than drift")

    for target in targets:
        panel, reason = asd.choose_panel(blocks, stem, target)
        result = asd.compute_reflectance(data, panel, target, panel_reflectance, EXCLUDE,
                                         MANUAL_IT_FACTOR, SPLICE_MODE, CORRECT_SWIR2)
        reflectance_results.append(result)
        print(f"  transect block {target['block_id']}: {len(result['labels'])} points, "
              f"panel = block {panel['block_id']} ({reason}, n={result['n_panel']})")
        for note in result["notes"]:
            print("      " + note)
        plots.plot_reflectance(result, asd.DEFAULT_MASK)
        plt.show()

if not reflectance_stems:
    print("No reflectance day detected. Set REFLECTANCE_STEMS if one of the stems is a "
          "panel-plus-transect day.")
    print("Stems present:", sorted(data.inventory["stem"].unique()))
