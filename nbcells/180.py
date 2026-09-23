data.inventory.drop(columns=["path"]).to_csv(OUTPUT_DIR / "inventory.csv", index=False)
block_table.to_csv(OUTPUT_DIR / "blocks.csv", index=False)
pd.concat([s.assign(block=bid) for bid, s in qc.items()], ignore_index=True) \
    .to_csv(OUTPUT_DIR / "qc_scores.csv", index=False)

if results:
    spectral = pd.DataFrame({"wavelength_nm": WL})
    for result in results:
        spectral[result["label"]] = np.where(result["masked"], np.nan, result["albedo"])
    spectral.to_csv(OUTPUT_DIR / "spectral_albedo.csv", index=False)
    asd.splice_qc_table(results).to_csv(OUTPUT_DIR / "albedo_qc.csv", index=False)
if broadband_table is not None:
    broadband_table.to_csv(OUTPUT_DIR / "broadband_albedo.csv", index=False)
for result in reflectance_results:
    frame = pd.DataFrame(result["reflectance"].T,
                         columns=[f"pt_{i:03d}" for i in result["labels"]])
    frame.insert(0, "wavelength_nm", WL)
    frame.to_csv(OUTPUT_DIR / f"reflectance_{result['label'].replace(' ', '_')}.csv", index=False)

print("wrote:", *sorted(p.name for p in OUTPUT_DIR.iterdir()), sep="\n  ")
