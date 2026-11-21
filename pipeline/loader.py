"""Helper for downstream teams (M2, M3): load a frozen feature table with checksum verification.

    from pipeline.loader import load_feature_table
    df, features, targets = load_feature_table("data/processed/h168_v0.2.0")
    train = df[df["split"] == "train"]
    X_train, y_train = train[features], train["target_ci_lifecycle"]

Rows are UTC-indexed hourly. `split` is train, validate, test, or none. Rows with
missing features (for example, the first weeks of 2019) are kept; drop them per model.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

TARGETS = ["target_ci_lifecycle", "target_ci_combustion"]


def load_feature_table(table_dir, verify=True):
    table_dir = Path(table_dir)
    manifest = json.loads((table_dir / "feature_table.manifest.json").read_text())
    parquet = table_dir / "feature_table.parquet"
    if verify:
        h = hashlib.sha256(parquet.read_bytes()).hexdigest()
        if h != manifest["sha256"]:
            raise ValueError(f"checksum mismatch for {parquet}: file {h[:12]} vs manifest {manifest['sha256'][:12]}")
    df = pd.read_parquet(parquet)
    return df, list(manifest["features"]), list(TARGETS)
