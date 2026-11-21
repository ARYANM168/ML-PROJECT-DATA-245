"""Baseline models and evaluation for the lifecycle and combustion-only targets.

Both targets use the same features and the same model settings. Baselines are
persistence and, for the 7-day model, a 4-week same-hour average.

Two evaluation modes:
- fixed split: fit once on train, score validate and test.
- rolling origin: weekly origins across the test period, refit on everything whose
  target is known before each origin, score the 168 hours after the origin.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    d2_absolute_error_score,
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_log_error,
)

from .splits import rolling_origins

HYDRO_FLAG = "ci_hydro_missing"


def _model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)


def _metrics(y, p):
    """Regression metrics on one set of predictions (scikit-learn definitions).

    MAE and RMSE are the headline metrics. The others are scikit-learn regression metrics:
    R2, explained variance, median absolute error, D2 absolute error, MAPE, max error, MSE,
    RMSLE, and mean gamma deviance. The scaled error (MASE) needs training data, so it is
    computed by pipeline/model.py naive_scale and add_mase, not here.

    The log-based and gamma metrics need non-negative predictions. Predictions are clipped to
    1e-6 for those metrics only, and the clipped count is reported as "n_clipped".
    """
    yt = np.asarray(y, dtype=float)
    yp = np.asarray(p, dtype=float)
    err = yt - yp
    yp_pos = np.clip(yp, 1e-6, None)
    return {
        "n": int(len(err)),
        "n_clipped": int(np.sum(yp < 1e-6)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mse": float(mean_squared_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "explained_variance": float(explained_variance_score(yt, yp)),
        "medae": float(median_absolute_error(yt, yp)),
        "d2_absolute_error": float(d2_absolute_error_score(yt, yp)),
        "mape": float(mean_absolute_percentage_error(yt, yp)),
        "max_error": float(np.max(np.abs(err))),
        "rmsle": float(root_mean_squared_log_error(yt, yp_pos)),
        "mean_gamma_deviance": float(mean_gamma_deviance(yt, yp_pos)),
    }


def naive_scale(panel, target, lag_hours=168):
    """MASE scale: mean |y(t) - y(t - lag)| over training rows where both are observed.

    Uses training rows only, so no test information enters the scale (Hyndman and Koehler, 2006).
    """
    train = panel[(panel["split"] == "train") & panel[target].notna()]
    lagged = panel[target].reindex(train.index - pd.Timedelta(hours=lag_hours))
    diff = np.abs(train[target].to_numpy() - lagged.to_numpy())
    diff = diff[~np.isnan(diff)]
    return float(diff.mean())


def add_mase(block, scale):
    """Add MASE (MAE divided by the naive scale) to every metric dict in a rolling result."""
    block["mase_scale_train_naive168_mae"] = scale
    for key, metrics in block.items():
        if isinstance(metrics, dict) and "mae" in metrics:
            metrics["mase"] = metrics["mae"] / scale
    return block


def fit_and_score(panel, target, baselines, feature_cols, split_col="split"):
    """Fit on train; score validate and test. Rows need a non-missing target and all features."""
    usable = panel[[target] + list(feature_cols)].notna().all(axis=1)
    model = _model()
    train = panel[usable & (panel[split_col] == "train")]
    model.fit(train[feature_cols], train[target])

    out = {"train_rows": int(len(train))}
    for split in ("validate", "test"):
        part = panel[usable & (panel[split_col] == split)]
        pred = model.predict(part[feature_cols])
        y = part[target].to_numpy()
        out[split] = {"model": _metrics(y, pred)}
        for name, col in baselines.items():
            base = panel.loc[part.index, col].to_numpy()
            ok = ~np.isnan(base)
            out[split][name] = _metrics(y[ok], base[ok])
        if split == "test" and HYDRO_FLAG in panel.columns:
            present = ~part[HYDRO_FLAG].to_numpy(dtype=bool)
            out[split]["model_hydro_present_rows"] = _metrics(y[present], pred[present])
        if split == "test" and "wx_any_imputed" in panel.columns:
            observed = ~part["wx_any_imputed"].to_numpy(dtype=bool)
            out[split]["model_weather_observed_rows"] = _metrics(y[observed], pred[observed])
    return out


def rolling_origin_eval(panel, target, baselines, feature_cols, start, end):
    """Weekly origins from `start` to `end`; refit at each origin on targets known before it."""
    usable = panel[[target] + list(feature_cols)].notna().all(axis=1)
    folds = rolling_origins(panel.index, start, end)

    ys, preds = [], []
    base_vals = {name: [] for name in baselines}
    for _, f in folds.iterrows():
        origin, fend = f["origin"], f["forecast_end"]
        train = panel[usable & (panel.index < origin)]
        test = panel[usable & (panel.index >= origin) & (panel.index <= fend)]
        if test.empty or train.empty:
            continue
        model = _model()
        model.fit(train[feature_cols], train[target])
        ys.append(test[target].to_numpy())
        preds.append(model.predict(test[feature_cols]))
        for name, col in baselines.items():
            base_vals[name].append(panel.loc[test.index, col].to_numpy())

    y = np.concatenate(ys)
    p = np.concatenate(preds)
    out = {"folds": int(len(ys)), "model": _metrics(y, p)}
    for name in baselines:
        b = np.concatenate(base_vals[name])
        ok = ~np.isnan(b)
        out[name] = _metrics(y[ok], b[ok])
    return out
