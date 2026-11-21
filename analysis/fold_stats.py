"""Fold-level analysis of the 7-day rolling-origin evaluation.

Reproduces the rolling-origin evaluation in pipeline/model.py, but keeps the per-fold
results. For each target it reports:
- per-fold MAE for the model and for the seasonal 4-week baseline
- how many of the 74 folds the model beats the seasonal baseline on
- the mean per-fold MAE difference (model minus seasonal), with a 95% bootstrap
  interval that resamples folds (folds are contiguous, so resampling hours would
  overstate precision)

Writes analysis/fold_stats.json and termreport/figures/fig_fold_mae.pdf.

    python3 -m analysis.fold_stats
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.loader import load_feature_table
from pipeline.model import _metrics, _model
from pipeline.run_pipeline import LEAD_WEEK, ROLLING_END, ROLLING_START, baselines_for
from pipeline.splits import rolling_origins

TABLE = "data/processed/h168_v0.3.0"
OUT_JSON = Path("analysis/fold_stats.json")
OUT_FIG = Path("termreport/figures/fig_fold_mae.pdf")
N_BOOT = 5000
SEED = 0


def per_fold(panel, target, baselines, features):
    usable = panel[[target] + list(features)].notna().all(axis=1)
    folds = rolling_origins(panel.index, ROLLING_START, ROLLING_END)
    rows = []
    for _, f in folds.iterrows():
        origin, fend = f["origin"], f["forecast_end"]
        train = panel[usable & (panel.index < origin)]
        test = panel[usable & (panel.index >= origin) & (panel.index <= fend)]
        if test.empty or train.empty:
            continue
        model = _model()
        model.fit(train[features], train[target])
        y = test[target].to_numpy()
        pred = model.predict(test[features])
        rec = {"origin": str(origin.date()), "n": int(len(y)),
               "model_mae": _metrics(y, pred)["mae"]}
        for name, col in baselines.items():
            b = panel.loc[test.index, col].to_numpy()
            ok = ~np.isnan(b)
            rec[f"{name}_mae"] = _metrics(y[ok], b[ok])["mae"]
        rows.append(rec)
    return rows


def bootstrap_ci(diffs, rng):
    means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(N_BOOT)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    df, features, _ = load_feature_table(TABLE)
    out = {"table": TABLE, "folds_expected": None, "targets": {}}
    fig_series = {}
    for tgt, kind in [("target_ci_lifecycle", "lifecycle"), ("target_ci_combustion", "combustion")]:
        baselines = baselines_for(kind, LEAD_WEEK)
        missing = [c for c in baselines.values() if c not in df.columns]
        if missing:
            raise SystemExit(f"baseline columns missing from table: {missing}")
        rows = per_fold(df, tgt, baselines, features)
        model = np.array([r["model_mae"] for r in rows])
        seas = np.array([r["seasonal_4w_mae"] for r in rows])
        diff = model - seas
        lo, hi = bootstrap_ci(diff, np.random.default_rng(SEED))
        out["targets"][tgt] = {
            "folds": len(rows),
            "scored_hours": int(sum(r["n"] for r in rows)),
            "fold_mae_model_median": float(np.median(model)),
            "fold_mae_model_min": float(model.min()),
            "fold_mae_model_max": float(model.max()),
            "fold_mae_seasonal_median": float(np.median(seas)),
            "folds_model_beats_seasonal": int((diff < 0).sum()),
            "folds_model_ties_or_loses": int((diff >= 0).sum()),
            "mean_fold_diff_model_minus_seasonal": float(diff.mean()),
            "bootstrap_95ci_mean_fold_diff": [lo, hi],
            "bootstrap_resamples": N_BOOT,
            "per_fold": rows,
        }
        fig_series[kind] = (model, seas)
        print(kind, {k: v for k, v in out["targets"][tgt].items() if k != "per_fold"})

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    m, s = fig_series["lifecycle"]
    x = np.arange(1, len(m) + 1)
    ax.plot(x, s, color="#8faadc", lw=1.0, label="Seasonal 4-week")
    ax.plot(x, m, color="#1f4e79", lw=1.0, label="GBM model")
    ax.set_xlabel("Weekly fold (1 to 74)")
    ax.set_ylabel("MAE per fold")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)


if __name__ == "__main__":
    main()
