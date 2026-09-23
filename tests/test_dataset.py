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
