results = [asd.compute_albedo(data, blocks[r], blocks[t], exclude=EXCLUDE,
                              manual_it_factor=MANUAL_IT_FACTOR, splice_mode=SPLICE_MODE,
                              correct_swir2=CORRECT_SWIR2, snr_min=SNR_MIN)
           for r, t in pairs]

if results:
    qc_table = asd.splice_qc_table(results)
    display(qc_table)
    lo, hi = asd.SPLICE_FACTOR_OK
    suspect = qc_table[(qc_table["vnir_splice_factor"] < lo)
                       | (qc_table["vnir_splice_factor"] > hi)]
    if len(suspect):
        print(f"\nSplice factor outside {lo} to {hi}. Check integration times and field notes:")
        display(suspect[["set", "it_factor", "raw_splice_step_pct", "vnir_splice_factor"]])

    plots.plot_albedo(results, SPLICE_MODE)
    plt.show()
else:
    print("No albedo pairs. If this is a reflectance-only day, go to section 5.")
