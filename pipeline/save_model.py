"""Save the headline models to disk for a quick demo, and load them back.

    python3 -m pipeline.save_model data/processed/h168_v0.3.0 --out models/h168_v0.3.0

Writes one joblib file per target plus model_card.json. The models are fit on every row
that has a target and all features (train, validate, and test splits together), so they
are the "latest" models for demonstration. They are NOT the evaluation results: the
reported numbers come from the rolling-origin evaluation in pipeline/model.py, which
never trains on the period it scores. The model card says this as well.

The pickle depends on the scikit-learn and joblib versions, so the model card records
them. Load the model in the same versions, or refit with pipeline.save_model.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .loader import TARGETS, load_feature_table
from .model import _model

TARGET_SHORT = {"target_ci_lifecycle": "lifecycle", "target_ci_combustion": "combustion"}


def fit_model(df, features, target):
    """Fit the same model settings used in the evaluation, on all usable rows."""
    usable = df[[target] + list(features)].notna().all(axis=1)
    model = _model()
    model.fit(df.loc[usable, features], df.loc[usable, target])
    return model, int(usable.sum())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_models(table_dir, out_dir):
    """Fit both headline targets on the table at `table_dir` and save them to `out_dir`."""
    df, features, targets = load_feature_table(table_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    card = {
        "purpose": "Demo models. Fit on all usable rows; not an evaluation result. "
                   "Evaluation numbers are in the table manifest (rolling origin).",
        "source_table": str(table_dir),
        "source_table_sha256": _sha256(Path(table_dir) / "feature_table.parquet"),
        "features": features,
        "model": "sklearn.ensemble.HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)",
        "library_versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                             "joblib": joblib.__version__},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {},
    }
    for target in targets:
        model, n_rows = fit_model(df, features, target)
        path = out_dir / f"ci_{TARGET_SHORT[target]}_h168.joblib"
        joblib.dump(model, path, compress=3)
        card["models"][target] = {
            "file": path.name,
            "file_sha256": _sha256(path),
            "file_bytes": path.stat().st_size,
            "training_rows": n_rows,
            "training_first_timestamp": str(df.index.min()),
            "training_last_timestamp": str(df.index.max()),
        }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2))
    return card


def load_model(path):
    """Load a model saved by save_models.

    SECURITY: joblib files are pickles, and unpickling can run arbitrary code. Load only
    files this pipeline wrote (check file_sha256 in model_card.json first). Never load a
    model file from an untrusted source.
    """
    return joblib.load(path)


def verify_model_file(path, expected_sha256):
    """Refuse to load a model file whose checksum does not match the model card."""
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"checksum mismatch for {path}: {actual[:12]} vs card {expected_sha256[:12]}")
    return load_model(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    card = save_models(args.table_dir, args.out)
    for target, info in card["models"].items():
        print(f"{target}: {info['file']} ({info['file_bytes']} bytes, {info['training_rows']} rows)")
    print(f"model card: {Path(args.out) / 'model_card.json'}")


if __name__ == "__main__":
    main()
