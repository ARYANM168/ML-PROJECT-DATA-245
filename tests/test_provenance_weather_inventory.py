import json
import os
import stat

import numpy as np
import pandas as pd
import pytest

from pipeline.inventory import inventory
from pipeline.provenance import make_read_only, record_download, sha256_of, verify_manifest
from pipeline.weather import area_weighted_mean, fill_with_climatology


# ---- provenance ----

def test_sha256_matches_known_value(tmp_path):
    f = tmp_path / "raw.csv"
    f.write_bytes(b"abc")
    assert sha256_of(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_record_download_makes_read_only_and_manifest_verifies(tmp_path):
    raw = tmp_path / "eia930.csv"
    raw.write_text("a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    rec = record_download(raw, "https://example.invalid/eia930", pudl_version="0.0.0-test",
                          manifest_path=manifest)
    assert rec["sha256"] == sha256_of(raw)
    mode = stat.S_IMODE(os.stat(raw).st_mode)
    assert not (mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    assert verify_manifest(manifest, tmp_path) == []


def test_verify_detects_changed_file(tmp_path):
    raw = tmp_path / "x.csv"
    raw.write_text("1\n")
    manifest = tmp_path / "m.json"
    record_download(raw, "src", manifest_path=manifest)
    os.chmod(raw, 0o644)
    raw.write_text("2\n")
    assert verify_manifest(manifest, tmp_path) == [("x.csv", "checksum mismatch")]


def test_manifest_replaces_record_for_same_file(tmp_path):
    raw = tmp_path / "y.csv"
    raw.write_text("1\n")
    manifest = tmp_path / "m.json"
    record_download(raw, "src1", manifest_path=manifest)
    os.chmod(raw, 0o644)
    raw.write_text("1\n2\n")
    record_download(raw, "src2", manifest_path=manifest)
    records = json.loads(manifest.read_text())
    assert len(records) == 1 and records[0]["source_url"] == "src2"


# ---- inventory ----

def test_inventory_counts_missing_negative_and_above_nameplate():
    ts = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({
        "ba": ["CISO"] * 4,
        "fuel": ["NG"] * 4,
        "timestamp": ts,
        "mw": [100.0, np.nan, -5.0, 250.0],
        "nameplate_mw": [200.0] * 4,
    })
    row = inventory(df).iloc[0]
    assert row["expected_hours"] == 4
    assert row["missing_hours"] == 1
    assert row["negative_values"] == 1
    assert row["above_nameplate"] == 1


# ---- weather ----

def test_area_weighted_mean_normalizes_over_present_stations():
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    vals = pd.DataFrame({"s1": [10.0, np.nan], "s2": [20.0, 40.0]}, index=idx)
    weights = pd.DataFrame({"CISO": [0.25, 0.75]}, index=["s1", "s2"])
    out = area_weighted_mean(vals, weights)
    assert out["CISO"].iloc[0] == pytest.approx((10 * 0.25 + 20 * 0.75) / 1.0)
    # hour 2: only s2 present, so its value stands alone
    assert out["CISO"].iloc[1] == pytest.approx(40.0)


def test_climatology_fill_uses_only_reference_period_and_flags():
    idx = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="h", tz="UTC")
    s = pd.Series(10.0, index=idx)           # reference years are all 10
    s.loc["2024-06-02":"2024-06-30"] = 1e6   # huge test-period values that must not leak in
    s.loc["2024-06-01"] = np.nan              # gap inside the test period
    filled, flag = fill_with_climatology(s, reference_end="2024-01-01")
    assert flag.loc["2024-06-01"].all()
    assert not flag.loc["2023-01-01"].any()
    assert (filled.loc["2024-06-01"] == 10.0).all()
