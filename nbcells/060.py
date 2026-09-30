display(data.inventory.groupby(["stem", "type", "it_ms", "role"], as_index=False)
        .agg(n=("file", "size"), first=("time", "min"), last=("time", "max"),
             swir_vis=("swir_vis_ratio", "mean"),
             swir1_gain=("swir1_gain", "first"), swir1_offset=("swir1_offset", "first")))

blocks = asd.find_blocks(data)
block_table = asd.block_table(blocks)
display(block_table)

asd.check_blocks(blocks)

reflectance_stems = REFLECTANCE_STEMS or asd.detect_reflectance_stems(blocks)
auto, orphans = asd.auto_pairs(blocks, skip_stems=reflectance_stems, group_by=PAIR_GROUP_BY)

pairs = asd.merge_pairs(auto,
                        manual=asd.resolve_pairs(blocks, MANUAL_PAIRS),
                        drop=asd.resolve_pairs(blocks, DROP_PAIRS))
paired = {b for pair in pairs for b in pair}
orphans = [o for o in orphans if o not in paired]

if reflectance_stems:
    print("reflectance days, handled in section 5:", reflectance_stems)
if pairs:
    display(pd.DataFrame([
        dict(reference=blocks[r]["key"], target=blocks[t]["key"],
             n_ref=blocks[r]["n"], n_tgt=blocks[t]["n"],
             source="manual" if (r, t) in set(asd.resolve_pairs(blocks, MANUAL_PAIRS))
             else "auto")
        for r, t in pairs]))
else:
    print("no albedo pairs")
if orphans:
    print(f"\n{len(orphans)} unpaired block(s). A calibrated irradiance set with no calibrated "
          f"counterpart is normal; anything else is worth checking.")
    display(asd.block_table({o: blocks[o] for o in orphans})
            [["block", "key", "role", "n", "type", "it_ms", "start"]])
