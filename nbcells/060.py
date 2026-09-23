display(data.inventory.groupby(["stem", "type", "it_ms", "role"], as_index=False)
        .agg(n=("file", "size"), first=("time", "min"), last=("time", "max"),
             swir_vis=("swir_vis_ratio", "mean"),
             swir1_gain=("swir1_gain", "first"), swir1_offset=("swir1_offset", "first")))

blocks = asd.find_blocks(data)
block_table = asd.block_table(blocks)
display(block_table)

reflectance_stems = REFLECTANCE_STEMS or asd.detect_reflectance_stems(blocks)
pairs, orphans = asd.auto_pairs(blocks, skip_stems=reflectance_stems)
# override by hand if a set is split across stems, e.g. pairs = [(0, 1), (2, 4)]

if reflectance_stems:
    print("reflectance days, handled in section 5:", reflectance_stems)
print("albedo pairs (reference block, target block):", pairs)
for bid in orphans:
    b = blocks[bid]
    print(f"   unpaired: block {bid}  {b['stem']}  {b['role']}  n={b['n']}  "
          f"{b['t_start']:%H:%M:%S}  it={b['it_ms']} ms")
