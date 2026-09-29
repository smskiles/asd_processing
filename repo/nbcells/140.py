rows, n_clamped = [], 0
for result in results:
    weight = weighting_for(result)
    if weight["E"] is None:
        print(f"{result['label']}: no irradiance weighting available, skipped")
        continue
    n_clamped += bool(weight["clamped"])
    bands = asd.broadband_albedo(WL, result["albedo"], weight["E"])
    rows.append(dict(set=result["label"],
                     time=result["time"].strftime("%Y-%m-%d %H:%M") if result["time"] else "",
                     weighting=weight["source"], sza=weight["sza"], column=weight["column"],
                     clamped=weight["clamped"],
                     splice_factor=round(result["splice1_factor"], 4),
                     **{k: round(v, 4) for k, v in bands.items()}))

broadband_table = pd.DataFrame(rows) if rows else None
if broadband_table is not None:
    display(broadband_table)
    if n_clamped:
        print(f"\n{n_clamped} set(s) fall outside the zenith-angle range of the table "
              f"({min(model.angles):g} to {max(model.angles):g} deg) and were clamped to the "
              f"nearest column. Near solar noon in spring at mid-latitude this is expected.")
