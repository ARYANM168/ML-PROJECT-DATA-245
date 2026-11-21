import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from pipeline.save_model import fit_model, load_model, save_models, verify_model_file


def _synthetic_table(tmp_path):
    """A small feature table with the same layout as the real ones (parquet + manifest)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC")
    df = pd.DataFrame({
        "f1": rng.normal(size=len(idx)),
        "f2": rng.normal(size=len(idx)),
        "target_ci_lifecycle": rng.normal(size=len(idx)) * 10 + 300,
        "target_ci_combustion": rng.normal(size=len(idx)) * 10 + 400,
        "split": "train",
    }, index=idx)
    df.loc[df.index[:20], "f1"] = np.nan  # rows with missing features must be skipped
    df.loc[df.index[-5:], "target_ci_lifecycle"] = np.nan
    table = tmp_path / "table"
    table.mkdir()
    parquet = table / "feature_table.parquet"
    df.to_parquet(parquet)
    manifest = {"sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(), "features": ["f1", "f2"]}
    (table / "feature_table.manifest.json").write_text(json.dumps(manifest))
    return table, df


def test_fit_model_skips_rows_with_missing_features_or_target(tmp_path):
    _, df = _synthetic_table(tmp_path)
    _, n_rows = fit_model(df, ["f1", "f2"], "target_ci_lifecycle")
    expected = int(df[["f1", "f2", "target_ci_lifecycle"]].notna().all(axis=1).sum())
    assert n_rows == expected == 400 - 20 - 5


def test_save_and_load_round_trip_gives_same_predictions(tmp_path):
    table, df = _synthetic_table(tmp_path)
    out = tmp_path / "models"
    card = save_models(table, out)

    assert set(card["models"]) == {"target_ci_lifecycle", "target_ci_combustion"}
    info = card["models"]["target_ci_lifecycle"]
    path = out / info["file"]
    assert path.exists() and info["file_bytes"] == path.stat().st_size

    in_memory, _ = fit_model(df, ["f1", "f2"], "target_ci_lifecycle")
    loaded = load_model(path)
    X = df[["f1", "f2"]].fillna(0.0)
    assert np.allclose(in_memory.predict(X), loaded.predict(X))


def test_model_card_records_checksums_and_features(tmp_path):
    table, _ = _synthetic_table(tmp_path)
    out = tmp_path / "models"
    card = save_models(table, out)
    on_disk = json.loads((out / "model_card.json").read_text())
    assert on_disk["features"] == ["f1", "f2"]
    assert "not an evaluation result" in on_disk["purpose"]
    for info in on_disk["models"].values():
        assert hashlib.sha256((out / info["file"]).read_bytes()).hexdigest() == info["file_sha256"]
    assert card["source_table_sha256"] == on_disk["source_table_sha256"]


def test_verify_model_file_refuses_wrong_checksum(tmp_path):
    table, _ = _synthetic_table(tmp_path)
    out = tmp_path / "models"
    card = save_models(table, out)
    info = card["models"]["target_ci_lifecycle"]
    path = out / info["file"]
    assert verify_model_file(path, info["file_sha256"]) is not None
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_model_file(path, "0" * 64)
