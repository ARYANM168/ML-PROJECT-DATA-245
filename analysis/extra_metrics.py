"""Full metric set for the 7-day rolling-origin evaluation.

Refits the model at each of the 74 weekly origins, exactly as pipeline/model.py does, pools
the predictions over the test period, and reports every metric in pipeline/model.py
(MAE, RMSE, R2, median absolute error, MAPE, max error) for the model and both baselines.

It also reports MASE, the scaled error from Hyndman and Koehler (2006). The scale is the
mean absolute error of the weekly naive forecast (value 168 hours earlier) on the training
split only, so no test information enters the scale.

Writes analysis/extra_metrics.json.

    python3 -m analysis.extra_metrics
"""

import json
from pathlib import Path

import numpy as np

from pipeline.loader import load_feature_table
from pipeline.model import _metrics, _model
from pipeline.run_pipeline import LEAD_WEEK, ROLLING_END, ROLLING_START, baselines_for
from pipeline.splits import rolling_origins

TABLE = "data/processed/h168_v0.3.0"
OUT = Path("analysis/extra_metrics.json")


def mase_scale(panel, target):
    """Mean |y(t) - y(t - 168h)| over training rows where both values are observed."""
    train = panel[(panel["split"] == "train") & panel[target].notna()]
    lagged = panel[target].reindex(train.index - np.timedelta64(LEAD_WEEK, "h"))
    diff = np.abs(train[target].to_numpy() - lagged.to_numpy())
    diff = diff[~np.isnan(diff)]
    return float(diff.mean())


def pooled_predictions(panel, target, baselines, features):
    usable = panel[[target] + list(features)].notna().all(axis=1)
    folds = rolling_origins(panel.index, ROLLING_START, ROLLING_END)
    ys, preds, base = [], [], {k: [] for k in baselines}
    for _, f in folds.iterrows():
        origin, fend = f["origin"], f["forecast_end"]
        train = panel[usable & (panel.index < origin)]
        test = panel[usable & (panel.index >= origin) & (panel.index <= fend)]
        if test.empty or train.empty:
            continue
        model = _model()
        model.fit(train[features], train[target])
        ys.append(test[target].to_numpy())
        preds.append(model.predict(test[features]))
        for k, col in baselines.items():
            base[k].append(panel.loc[test.index, col].to_numpy())
    return np.concatenate(ys), np.concatenate(preds), {k: np.concatenate(v) for k, v in base.items()}


def main():
    panel, features, _ = load_feature_table(TABLE)
    out = {"table": TABLE, "test_window": [ROLLING_START, ROLLING_END], "targets": {}}
    for tgt, kind in [("target_ci_lifecycle", "lifecycle"), ("target_ci_combustion", "combustion")]:
        baselines = baselines_for(kind, LEAD_WEEK)
        y, p, base = pooled_predictions(panel, tgt, baselines, features)
        scale = mase_scale(panel, tgt)
        block = {"scale_train_naive168_mae": scale, "model": _metrics(y, p)}
        block["model"]["mase"] = block["model"]["mae"] / scale
        for k in baselines:
            ok = ~np.isnan(base[k])
            m = _metrics(y[ok], base[k][ok])
            m["mase"] = m["mae"] / scale
            block[k] = m
        out["targets"][tgt] = block
        print(tgt, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in block.items() if k != "scale_train_naive168_mae"})
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
