import datetime

import numpy as np
import pytest

from asdspec import (auto_pairs, detect_reflectance_stems, find_blocks, scan_folder)
from conftest import write_asd


def build_albedo_day(root, solar, snowy):
    """Ten reference then ten target files, with a pause while the sensor is flipped."""
    root.mkdir(parents=True, exist_ok=True)
    t = datetime.datetime(2024, 5, 24, 14, 3, 0)
    for i in range(10):
        write_asd(root / f"240524a.{i:03d}", solar, t + datetime.timedelta(seconds=3 * i))
    t2 = t + datetime.timedelta(seconds=150)
    for i in range(10):
        write_asd(root / f"240524a.{10 + i:03d}", solar * snowy,
                  t2 + datetime.timedelta(seconds=3 * i))


def build_transect_day(root, solar, snowy):
    """Opening panel, a longer transect at a different integration time, closing panel."""
    root.mkdir(parents=True, exist_ok=True)
    t = datetime.datetime(2026, 8, 29, 15, 10, 0)
    for i in range(10):
        write_asd(root / f"260829r5.{i:03d}", solar, t + datetime.timedelta(seconds=3 * i))
    t2 = t + datetime.timedelta(seconds=60)
    for i in range(21):
        write_asd(root / f"260829r5.{10 + i:03d}", 2 * solar * snowy * (0.6 + 0.04 * i),
                  t2 + datetime.timedelta(seconds=7 * i), it_ms=136)
    t3 = t2 + datetime.timedelta(seconds=200)
    for i in range(10):
        write_asd(root / f"260829r5.{31 + i:03d}", solar,
                  t3 + datetime.timedelta(seconds=3 * i), swir1_offset=1800)


def test_roles_and_blocks_on_an_albedo_day(tmp_path, solar, snowy):
    build_albedo_day(tmp_path, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    assert len(data) == 20
    blocks = find_blocks(data)
    assert len(blocks) == 2
    roles = [b["role"] for b in blocks.values()]
    assert roles == ["reference", "target"]
    assert all(b["n"] == 10 for b in blocks.values())

    pairs, orphans = auto_pairs(blocks)
    assert pairs == [(0, 1)]
    assert orphans == []
    assert detect_reflectance_stems(blocks) == []


def test_transect_day_splits_into_panel_transect_panel(tmp_path, solar, snowy):
    build_transect_day(tmp_path, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    assert [b["n"] for b in blocks.values()] == [10, 21, 10]
    assert [b["role"] for b in blocks.values()] == ["reference", "target", "reference"]
    # the closing panel was re-optimized, so it must be its own block
    assert blocks[0]["swir1_offset"] != blocks[2]["swir1_offset"]
    assert detect_reflectance_stems(blocks) == ["260829r5"]
    pairs, _ = auto_pairs(blocks, skip_stems=detect_reflectance_stems(blocks))
    assert pairs == []


def test_a_target_run_is_not_split_by_a_few_odd_scans(tmp_path, solar, snowy):
    """Short runs of the minority role inside a long run are transitional scans."""
    build_transect_day(tmp_path, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    target = [b for b in blocks.values() if b["role"] == "target"][0]
    assert target["indices"] == list(range(10, 31))


def test_mean_respects_exclusions(tmp_path, solar, snowy):
    build_albedo_day(tmp_path, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    mean, kept = data.mean(blocks[0], exclude=[0, 1])
    assert len(kept) == 8
    with pytest.raises(ValueError, match="every spectrum excluded"):
        data.mean(blocks[0], exclude=list(range(10)))


def test_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_folder(tmp_path, verbose=False)


def test_multiple_date_folders_are_scanned_recursively(tmp_path, solar, snowy):
    build_albedo_day(tmp_path / "240411", solar, snowy)
    for path in sorted((tmp_path / "240411").iterdir()):
        path.rename(path.with_name(path.name.replace("240524a", "240411a")))
    build_albedo_day(tmp_path / "240524", solar, snowy)

    data = scan_folder(tmp_path, verbose=False)
    assert len(data) == 40
    blocks = find_blocks(data)
    assert len(blocks) == 4
    assert sorted({b["stem"] for b in blocks.values()}) == ["240411a", "240524a"]
    pairs, orphans = auto_pairs(blocks)
    assert len(pairs) == 2 and orphans == []


def test_identical_file_names_in_two_folders_do_not_overwrite(tmp_path, solar, snowy):
    build_albedo_day(tmp_path / "raw", solar, snowy)
    build_albedo_day(tmp_path / "copy", solar * 0.5, snowy)

    data = scan_folder(tmp_path, verbose=False)
    assert len(data) == 40
    assert len(data.spectra) == 40  # keyed by path, not by file name
    levels = sorted(np.nanmean(s) for s in data.spectra.values())
    assert levels[0] < levels[-1]  # the two copies survived separately


def test_exclusions_accept_block_id_or_stable_key(tmp_path, solar, snowy):
    from asdspec import resolve_exclusions
    build_albedo_day(tmp_path, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)

    by_key = resolve_exclusions(blocks, {blocks[1]["key"]: [3]})
    by_id = resolve_exclusions(blocks, {1: [3]})
    assert by_key == by_id == {1: [3]}
    assert blocks[1]["key"] == "240524a:10-19"
    with pytest.raises(KeyError):
        resolve_exclusions(blocks, {"nope:0-9": [1]})


def build_session(root, date, hour, minute, solar, snowy):
    """One session: uplooking, irradiance, downlooking, under three file stems."""
    root.mkdir(parents=True, exist_ok=True)
    t = datetime.datetime(2024, 4, 11, hour, minute, 0)
    for i in range(10):
        write_asd(root / f"{date}Au.{i:03d}", solar, t + datetime.timedelta(seconds=3 * i))
    for i in range(10):
        write_asd(root / f"{date}i.{i:03d}", solar,
                  t + datetime.timedelta(minutes=7, seconds=3 * i), data_type=4)
    for i in range(10):
        write_asd(root / f"{date}Ad.{i:03d}", solar * snowy,
                  t + datetime.timedelta(minutes=15, seconds=3 * i))


def test_same_stem_in_two_folders_stays_two_measurement_sets(tmp_path, solar, snowy):
    """Without folder awareness these interleave on file index into n=1 blocks."""
    build_session(tmp_path / "morning", "240411", 10, 19, solar, snowy)
    build_session(tmp_path / "afternoon", "240411", 12, 39, solar, snowy)

    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    assert len(blocks) == 6
    assert all(b["n"] == 10 for b in blocks.values())
    assert {b["folder"] for b in blocks.values()} == {"morning", "afternoon"}
    assert len({b["key"] for b in blocks.values()}) == 6


def test_pairs_across_stems_within_a_day(tmp_path, solar, snowy):
    """Uplooking and downlooking often carry different stems, such as Au and Ad."""
    build_session(tmp_path / "morning", "240411", 10, 19, solar, snowy)
    build_session(tmp_path / "afternoon", "240411", 12, 39, solar, snowy)

    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    pairs, orphans = auto_pairs(blocks, verbose=False)

    assert len(pairs) == 2
    for ref, tgt in pairs:
        assert blocks[ref]["stem"].endswith("Au")
        assert blocks[tgt]["stem"].endswith("Ad")
        assert blocks[ref]["folder"] == blocks[tgt]["folder"]
    # the calibrated irradiance sets have no calibrated target, so they stay unpaired
    assert {blocks[o]["stem"] for o in orphans} == {"240411i"}


def test_pairing_requires_matching_spectrum_type(tmp_path, solar, snowy):
    """The irradiance set sits closer in time than the uplooking one and must not win."""
    build_session(tmp_path, "240411", 10, 19, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    pairs, _ = auto_pairs(blocks, verbose=False)

    assert len(pairs) == 1
    ref, tgt = pairs[0]
    assert blocks[ref]["type"] == blocks[tgt]["type"] == "RAW"


def test_group_by_stem_still_available(tmp_path, solar, snowy):
    build_session(tmp_path, "240411", 10, 19, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    pairs, _ = auto_pairs(blocks, group_by="stem", verbose=False)
    assert pairs == []            # Au and Ad are different stems
    with pytest.raises(ValueError):
        auto_pairs(blocks, group_by="nonsense")


def test_distant_pairs_are_reported(tmp_path, solar, snowy, capsys):
    build_session(tmp_path, "240411", 10, 19, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    auto_pairs(blocks, warn_minutes=5, verbose=True)
    assert "minutes apart" in capsys.readouterr().out


def test_mismatched_types_refuse_to_make_an_albedo(tmp_path, solar, snowy):
    from asdspec import compute_albedo
    build_session(tmp_path, "240411", 10, 19, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)
    irradiance = [b for b in blocks.values() if b["type"] == "IRRADIANCE"][0]
    target = [b for b in blocks.values() if b["role"] == "target"][0]
    with pytest.raises(ValueError, match="not an albedo"):
        compute_albedo(data, irradiance, target)


def test_unstable_role_split_collapses_to_one_role(tmp_path, wl, capsys):
    """A ratio straddling the threshold must not fragment a set into slivers."""
    from conftest import write_asd
    vis = np.exp(-((wl - 500) / 300) ** 2)
    swir = np.exp(-((wl - 1600) / 150) ** 2)
    ratios = [0.13, 0.12, 0.14, 0.17, 0.18, 0.16, 0.19, 0.17, 0.18, 0.13,
              0.14, 0.12, 0.16, 0.18, 0.17, 0.19, 0.16, 0.13, 0.14, 0.12]
    t = datetime.datetime(2024, 4, 1, 12, 14, 0)
    for i, r in enumerate(ratios):
        write_asd(tmp_path / f"240401A.{i:03d}", 1000 * vis + 1000 * r * swir,
                  t + datetime.timedelta(seconds=3 * i))

    data = scan_folder(tmp_path, verbose=True)
    assert "unstable reference/target split" in capsys.readouterr().out
    blocks = find_blocks(data)
    assert len(blocks) == 1
    assert blocks[0]["n"] == 20


def test_check_blocks_flags_slivers(tmp_path, solar, snowy, capsys):
    from asdspec import check_blocks
    build_albedo_day(tmp_path, solar, snowy)
    data = scan_folder(tmp_path, verbose=False)
    blocks = find_blocks(data)

    assert check_blocks(blocks, verbose=False) == []
    tiny = {0: dict(blocks[0], n=1, indices=[0], files=blocks[0]["files"][:1],
                    paths=blocks[0]["paths"][:1])}
    assert len(check_blocks(tiny, verbose=True)) == 1
    assert "fewer than 3 files" in capsys.readouterr().out
