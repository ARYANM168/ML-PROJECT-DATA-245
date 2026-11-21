"""
Final carbon-intensity forecasting experiment.

This script runs the corrected temporal baselines and the three learned model
families required by the project:

  * Elastic Net
  * Random Forest
  * Gradient Boosting

The learned models are fitted separately for each forecast horizon and inside
each chronological monthly fold.  No future target values are used as model
features.  The script writes the fold-level results and the summary table that
should be used to update the final report.

Default run:
    python carbon_forecast_models_final.py

Smoke test on synthetic data:
    python carbon_forecast_models_final.py --data-source demo --demo-days 900

The default data source is the public PUDL EIA-930 Parquet file.  The first
PUDL run may take a while and creates caiso_hourly.csv for later reruns.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------- settings -----------------------------
DATA_SOURCE = "pudl"  # "pudl", "csv", or "demo"
PUDL_URL = (
    "s3://pudl.catalyst.coop/stable/"
    "core_eia930__hourly_net_generation_by_energy_source.parquet"
)
BA_CODE = "CISO"
LOCAL_TZ = "America/Los_Angeles"
CACHE_PATH = "caiso_hourly.csv"
CSV_PATH = "caiso_hourly.csv"
TIME_COL = "timestamp"
TARGET_COL = "carbon_intensity"
HORIZONS = [1, 6, 24]
MIN_TRAIN_MONTHS = 12
TOP_K = 6
WINDOW_HOURS = 3
MASE_LAG = 24
RANDOM_STATE = 42

# These are the same direct-combustion placeholder values used by the existing
# corrected baseline run. They are reported transparently in the PDF report.
EMISSION_FACTORS = [
    ("coal", 1000.0),
    ("oil", 800.0),
    ("gas", 450.0),
    ("other", 300.0),
    ("unknown", 300.0),
    ("nuclear", 0.0),
    ("hydro", 0.0),
    ("pumped", 0.0),
    ("solar", 0.0),
    ("wind", 0.0),
    ("geothermal", 0.0),
    ("battery", 0.0),
    ("storage", 0.0),
]

BASELINE_MODELS = [
    "persistence",
    "seasonal_naive_24",
    "seasonal_naive_168",
]
LEARNED_MODELS = ["elastic_net", "random_forest", "gradient_boosting"]
ALL_MODELS = BASELINE_MODELS + LEARNED_MODELS


# ----------------------------- data -----------------------------
def load_series(path: str | Path) -> pd.Series:
    """Load a timestamp/carbon-intensity CSV onto an exact hourly grid."""
    df = pd.read_csv(path, parse_dates=[TIME_COL]).sort_values(TIME_COL)
    s = df.set_index(TIME_COL)[TARGET_COL].astype(float)
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC").tz_localize(None)
    s = s[~s.index.duplicated(keep="first")]
    return s.asfreq("h")


def trim_to_first_valid(s: pd.Series) -> pd.Series:
    """Remove the leading empty section of the derived target series."""
    first_valid = s.first_valid_index()
    if first_valid is None:
        raise ValueError("The carbon-intensity series contains no valid values.")
    return s.loc[first_valid:]


def lag_by_time(s: pd.Series, hours: int) -> pd.Series:
    """Look up a value by elapsed timestamp, not by row position."""
    lagged = s.reindex(s.index - pd.Timedelta(hours=hours))
    lagged.index = s.index
    return lagged


def _factor(source: object) -> float:
    name = str(source).lower()
    for key, value in EMISSION_FACTORS:
        if key in name:
            return value
    return np.nan


def load_from_pudl() -> pd.Series:
    """Read CAISO EIA-930 generation by fuel and derive hourly intensity."""
    try:
        raw = pd.read_parquet(
            PUDL_URL,
            filters=[("balancing_authority_code_eia", "==", BA_CODE)],
            storage_options={"anon": True},
        )
    except ImportError as exc:
        raise SystemExit(
            f"A required Parquet/S3 package is missing ({exc}). "
            "Run: python -m pip install -r requirements_final_models.txt"
        ) from exc

    print(f"PUDL: {len(raw):,} rows for {BA_CODE}.")
    gen_cols = [
        column
        for column in raw.columns
        if "net_generation" in column and column.endswith("mwh")
    ]
    reported = next((c for c in gen_cols if "reported" in c), None)
    imputed = next((c for c in gen_cols if "imputed" in c), None)
    if reported is None and imputed is None:
        raise SystemExit(
            f"No net-generation column found in {gen_cols or list(raw.columns)}."
        )

    values = raw[imputed] if imputed else raw[reported]
    if imputed and reported:
        values = values.combine_first(raw[reported])

    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(raw["datetime_utc"]),
            "source": raw["generation_energy_source"],
            "mwh": values.astype(float).clip(lower=0),
        }
    )
    frame["factor"] = frame["source"].map(_factor)
    unmatched = frame.loc[frame["factor"].isna(), "source"].unique()
    if len(unmatched):
        print(
            "WARNING: unmatched fuel labels treated as zero: "
            f"{list(unmatched)}"
        )
        frame["factor"] = frame["factor"].fillna(0.0)

    frame["emissions"] = frame["mwh"] * frame["factor"]
    hourly = frame.groupby("timestamp_utc")[["emissions", "mwh"]].sum()
    ci = (hourly["emissions"] / hourly["mwh"]).where(hourly["mwh"] > 0)
    if ci.index.tz is None:
        ci.index = ci.index.tz_localize("UTC")
    ci.index = ci.index.tz_convert(LOCAL_TZ).tz_localize(None)
    ci = ci[~ci.index.duplicated(keep="first")].sort_index().asfreq("h")
    ci.name = TARGET_COL
    return ci


def make_demo_data(days: int = 900, seed: int = 0) -> pd.Series:
    """Create synthetic data only for a smoke test, never for report results."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=days * 24, freq="h")
    hour = index.hour.to_numpy()
    day_of_year = index.dayofyear.to_numpy()
    daily = -90 * np.exp(-((hour - 13) ** 2) / 18)
    yearly = 40 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
    noise = rng.normal(0, 15, len(index))
    return pd.Series(250 + daily + yearly + noise, index=index, name=TARGET_COL)


def load_target(data_source: str, force_refresh: bool, demo_days: int) -> pd.Series:
    if data_source == "demo":
        print("DATA_SOURCE=demo: synthetic smoke-test data only.")
        return make_demo_data(demo_days)

    if data_source == "csv":
        if not Path(CSV_PATH).exists():
            raise SystemExit(f"Cannot find {CSV_PATH} for DATA_SOURCE=csv.")
        return load_series(CSV_PATH)

    if data_source != "pudl":
        raise SystemExit("DATA_SOURCE must be pudl, csv, or demo.")

    if Path(CACHE_PATH).exists() and not force_refresh:
        print(f"Loading cached target from {CACHE_PATH}.")
        return load_series(CACHE_PATH)

    target = load_from_pudl()
    target.rename_axis(TIME_COL).to_csv(CACHE_PATH)
    print(f"Saved target cache to {CACHE_PATH}.")
    return target


# ----------------------------- features -----------------------------
def make_features(y: pd.Series, horizon: int) -> pd.DataFrame:
    """Build features available at the forecast origin for target time T.

    For a target at time T and horizon h, lag_0h means y[T-h], the latest
    observed target available at the forecast origin. Other lags are earlier
    observations. Calendar features are known for the target timestamp.
    """
    features = pd.DataFrame(index=y.index)
    origin_lags = [0, 1, 2, 3, 6, 12, 24, 48, 72, 168]
    for lag in origin_lags:
        features[f"lag_{lag}h"] = lag_by_time(y, horizon + lag)

    hours = y.index.hour.to_numpy()
    weekdays = y.index.dayofweek.to_numpy()
    day_of_year = y.index.dayofyear.to_numpy()
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * weekdays / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekdays / 7)
    features["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    features["is_weekend"] = (weekdays >= 5).astype(float)
    return features


def make_learned_model(name: str, n_estimators: int):
    """Return one fixed, reproducible model configuration."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is missing. Run: "
            "python -m pip install -r requirements_final_models.txt"
        ) from exc

    if name == "elastic_net":
        return make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=0.05,
                l1_ratio=0.5,
                max_iter=30000,
                random_state=RANDOM_STATE,
            ),
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            loss="squared_error",
        )
    raise ValueError(f"Unknown learned model: {name}")


# ----------------------------- baselines and metrics -----------------------------
def baseline_forecast(model: str, y: pd.Series, horizon: int) -> pd.Series:
    if model == "persistence":
        return lag_by_time(y, horizon)
    if model == "seasonal_naive_24":
        return lag_by_time(y, 24)
    if model == "seasonal_naive_168":
        return lag_by_time(y, 168)
    raise ValueError(model)


def _clean(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    return actual[valid], predicted[valid]


def mae(actual, predicted) -> float:
    actual, predicted = _clean(actual, predicted)
    return float(np.mean(np.abs(actual - predicted))) if len(actual) else np.nan


def rmse(actual, predicted) -> float:
    actual, predicted = _clean(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2))) if len(actual) else np.nan


def mase_scale(train: pd.Series, lag: int = MASE_LAG) -> float:
    values = train.dropna().to_numpy()
    if len(values) <= lag:
        return np.nan
    return float(np.mean(np.abs(values[lag:] - values[:-lag])))


def mase(actual, predicted, scale: float) -> float:
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return mae(actual, predicted) / scale


def decision_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"actual": actual, "predicted": predicted})
    hits, regrets, achieved, possible = [], [], [], []

    for _, day in frame.groupby(frame.index.date):
        if len(day) != 24 or day.isna().any().any():
            continue
        actual_values = day["actual"].to_numpy()
        predicted_values = day["predicted"].to_numpy()
        true_clean = set(np.argsort(actual_values)[:TOP_K])
        predicted_clean = set(np.argsort(predicted_values)[:TOP_K])
        hits.append(len(true_clean & predicted_clean) / TOP_K)

        starts = range(0, 24 - WINDOW_HOURS + 1)
        true_windows = np.array(
            [actual_values[s : s + WINDOW_HOURS].mean() for s in starts]
        )
        predicted_windows = np.array(
            [predicted_values[s : s + WINDOW_HOURS].mean() for s in starts]
        )
        chosen = int(np.argmin(predicted_windows))
        best = int(np.argmin(true_windows))
        regrets.append(true_windows[chosen] - true_windows[best])
        achieved.append(actual_values.mean() - true_windows[chosen])
        possible.append(actual_values.mean() - true_windows[best])

    if not hits or np.sum(possible) == 0:
        return {"top_k_hit": np.nan, "regret": np.nan, "capture": np.nan}
    return {
        "top_k_hit": float(np.mean(hits)),
        "regret": float(np.mean(regrets)),
        "capture": float(np.sum(achieved) / np.sum(possible)),
    }


# ----------------------------- weekly alignment output -----------------------------
def write_weekly_alignment_check(y: pd.Series) -> None:
    check = pd.DataFrame(
        {
            "actual": y,
            "yesterday": lag_by_time(y, 24),
            "last_week": lag_by_time(y, 168),
        }
    ).dropna()
    if check.empty:
        print("Weekly alignment check skipped: no complete lagged rows.")
        return

    sample = None
    for day in pd.Index(check.index.normalize()).drop_duplicates():
        candidate = check.loc[day : day + pd.Timedelta(days=13, hours=23)]
        if len(candidate) >= 24 * 12:
            sample = candidate
            break
    if sample is None:
        sample = check.iloc[: 24 * 14]

    sample.to_csv("weekly_alignment_check.csv", index_label=TIME_COL)
    try:
        import matplotlib.pyplot as plt

        ax = sample.plot(figsize=(12, 5), title="Weekly baseline alignment check")
        ax.set_xlabel("Local timestamp")
        ax.set_ylabel("Carbon intensity (gCO2e/kWh)")
        ax.figure.tight_layout()
        ax.figure.savefig("weekly_alignment_check.png", dpi=150)
        plt.close(ax.figure)
        print("Saved weekly_alignment_check.csv and weekly_alignment_check.png")
    except ImportError:
        print("Saved weekly_alignment_check.csv; matplotlib is unavailable for PNG.")


# ----------------------------- evaluation -----------------------------
def evaluate(y: pd.Series, n_estimators: int) -> pd.DataFrame:
    y = trim_to_first_valid(y)
    first_test = y.index.min().to_period("M") + MIN_TRAIN_MONTHS
    months = pd.period_range(first_test, y.index.max().to_period("M"), freq="M")
    rows = []
    print(f"Scheduled monthly test folds: {len(months)}")

    for fold_number, month in enumerate(months, start=1):
        start, end = month.start_time, month.end_time
        train = y.loc[: start - pd.Timedelta(hours=1)]
        test_index = y.loc[start:end].dropna().index
        if len(test_index) == 0:
            continue

        print(f"Fold {fold_number}/{len(months)}: {month} ({len(test_index):,} test hours)")
        scale = mase_scale(train)
        actual = y.reindex(test_index)

        for model_name in ALL_MODELS:
            for horizon in HORIZONS:
                if model_name == "seasonal_naive_24" and horizon > 24:
                    continue

                if model_name in BASELINE_MODELS:
                    predicted = baseline_forecast(model_name, y, horizon).reindex(test_index)
                else:
                    features = make_features(y, horizon)
                    train_index = y.index[(y.index < start) & y.notna()]
                    train_frame = features.loc[train_index].copy()
                    train_frame["target"] = y.loc[train_index]
                    train_frame = train_frame.dropna()
                    if train_frame.empty:
                        raise RuntimeError(
                            f"No complete training rows for {model_name}, "
                            f"horizon {horizon}, fold {month}."
                        )

                    test_features = features.reindex(test_index)
                    valid_test = test_features.notna().all(axis=1)
                    predicted = pd.Series(np.nan, index=test_index, dtype=float)
                    estimator = make_learned_model(model_name, n_estimators)
                    estimator.fit(
                        train_frame.drop(columns="target"),
                        train_frame["target"],
                    )
                    if valid_test.any():
                        predicted.loc[valid_test] = estimator.predict(
                            test_features.loc[valid_test]
                        )

                row = {
                    "month": str(month),
                    "model": model_name,
                    "horizon": horizon,
                    "mae": mae(actual, predicted),
                    "rmse": rmse(actual, predicted),
                    "mase": mase(actual, predicted, scale),
                }
                if horizon == 24:
                    row.update(decision_metrics(actual, predicted))
                else:
                    row.update(
                        {"top_k_hit": np.nan, "regret": np.nan, "capture": np.nan}
                    )
                rows.append(row)

    if not rows:
        raise RuntimeError("No evaluation rows were generated.")
    return pd.DataFrame(rows)


def write_summary(by_fold: pd.DataFrame) -> pd.DataFrame:
    summary = (
        by_fold.groupby(["model", "horizon"])[
            ["mae", "rmse", "mase", "top_k_hit", "regret", "capture"]
        ]
        .mean()
        .round(3)
    )
    summary.to_csv("results_all_models_summary.csv")
    Path("results_all_models_summary.txt").write_text(
        summary.to_string() + "\n", encoding="utf-8"
    )
    return summary


def write_metadata(y: pd.Series, by_fold: pd.DataFrame, n_estimators: int) -> None:
    with open("model_run_metadata.txt", "w", encoding="utf-8") as handle:
        handle.write("Final learned-model carbon-intensity run\n")
        handle.write("=========================================\n")
        handle.write(f"First valid target hour: {y.index.min()}\n")
        handle.write(f"Last target hour: {y.index.max()}\n")
        handle.write(f"Nonempty monthly folds: {by_fold['month'].nunique()}\n")
        handle.write(f"Horizons: {HORIZONS}\n")
        handle.write(f"Random seed: {RANDOM_STATE}\n")
        handle.write(f"Tree estimators: {n_estimators}\n")
        handle.write("Models: " + ", ".join(ALL_MODELS) + "\n")
        handle.write("Features: timestamp-safe target lags plus calendar sin/cos features\n")
        handle.write("Emission factors: direct-combustion placeholders documented in the report\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        choices=["pudl", "csv", "demo"],
        default=DATA_SOURCE,
        help="Use public PUDL data, an existing CSV, or synthetic smoke-test data.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore caiso_hourly.csv and download the PUDL target again.",
    )
    parser.add_argument(
        "--demo-days",
        type=int,
        default=900,
        help="Number of synthetic days for --data-source demo.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees for Random Forest and Gradient Boosting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_estimators < 10:
        raise SystemExit("--n-estimators must be at least 10.")

    target = trim_to_first_valid(
        load_target(args.data_source, args.force_refresh, args.demo_days)
    )
    print(f"First valid target hour: {target.index.min()}")
    write_weekly_alignment_check(target)

    by_fold = evaluate(target, args.n_estimators)
    by_fold.to_csv("results_all_models_by_fold.csv", index=False)
    summary = write_summary(by_fold)
    write_metadata(target, by_fold, args.n_estimators)

    print(f"\nNonempty test folds: {by_fold['month'].nunique()} months")
    print("\nSummary results:")
    print(summary)
    print("\nCreated:")
    print("  results_all_models_by_fold.csv")
    print("  results_all_models_summary.csv")
    print("  results_all_models_summary.txt")
    print("  model_run_metadata.txt")
    print("  weekly_alignment_check.csv")
    print("  weekly_alignment_check.png (when matplotlib is installed)")


if __name__ == "__main__":
    main()
