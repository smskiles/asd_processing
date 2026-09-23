qc = asd.qc_all(data, blocks, EXCLUDE, z_threshold=Z_THRESHOLD)

if AUTO_EXCLUDE_FLAGGED:
    for bid, scores in qc.items():
        auto = scores.loc[scores["flagged"], "index"].tolist()
        if auto:
            EXCLUDE[bid] = sorted(set(EXCLUDE.get(bid, [])) | set(auto))

for bid, block in blocks.items():
    scores = qc[bid]
    plots.plot_block_qc(data, block, scores, Z_THRESHOLD)
    plt.show()
    flagged = scores.loc[scores["flagged"], "index"].tolist()
    dropped = EXCLUDE.get(bid, [])
    print(f"   flagged: {flagged or 'none'}    excluded: {dropped or 'none'}")
