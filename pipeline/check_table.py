"""Loader check for downstream teams (M2, M3).

Opens a frozen feature table, verifies its SHA-256 against the manifest, checks the
feature list and targets, and prints what a downstream loader needs.

Usage:
    python -m pipeline.check_table data/processed/h168_v0.2.0
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

TARGETS = ["target_ci_lifecycle", "target_ci_combustion"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(table_dir):
    table_dir = Path(table_dir)
    manifest = json.loads((table_dir / "feature_table.manifest.json").read_text())
    parquet = table_dir / "feature_table.parquet"
    problems = []

    checksum = sha256_file(parquet)
    if checksum != manifest["sha256"]:
        problems.append(f"checksum mismatch: file {checksum[:12]} vs manifest {manifest['sha256'][:12]}")

    df = pd.read_parquet(parquet)
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        problems.append("index is not a timezone-aware DatetimeIndex (UTC expected)")
    elif str(df.index.tz) != "UTC":
        problems.append(f"index timezone is {df.index.tz}, expected UTC")
    if df.index.has_duplicates:
        problems.append("duplicate timestamps in the index")

    missing = [c for c in manifest["features"] if c not in df.columns]
    if missing:
        problems.append(f"missing feature columns: {missing}")
    for t in TARGETS:
        if t not in df.columns:
            problems.append(f"missing target column: {t}")

    n_feat = len(manifest["features"])
    if not 25 <= n_feat <= 30:
        problems.append(f"feature count {n_feat} outside 25-30")

    report = {
        "table": str(parquet),
        "sha256_ok": checksum == manifest["sha256"],
        "version": manifest["version"],
        "horizon_hours": manifest.get("horizon_hours"),
        "rows": int(len(df)),
        "index_start_utc": str(df.index.min()),
        "index_end_utc": str(df.index.max()),
        "feature_count": n_feat,
        "features": manifest["features"],
        "targets": TARGETS,
        "target_non_missing": {t: int(df[t].notna().sum()) for t in TARGETS if t in df.columns},
        "dtypes": {c: str(df[c].dtype) for c in manifest["features"] + TARGETS if c in df.columns},
        "no_future_check": manifest["no_future_check"]["result"],
        "problems": problems,
        "result": "PASS" if not problems else "FAIL",
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir")
    ap.add_argument("--save", default=None, help="optional path to write the JSON report")
    args = ap.parse_args()
    report = check(args.table_dir)
    text = json.dumps(report, indent=2)
    print(text)
    if args.save:
        Path(args.save).write_text(text)
    sys.exit(0 if report["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
