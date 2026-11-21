import numpy as np
import pandas as pd
import pytest

from pipeline.features import build_features


@pytest.fixture
def hourly():
    idx = pd.date_range("2023-01-01", periods=24 * 30, freq="h", tz="UTC")  # 720 hours
    rng = np.random.default_rng(0)
    ci = pd.Series(rng.uniform(100, 400, len(idx)), index=idx, name="ci")
    demand = pd.Series(rng.uniform(10_000, 20_000, len(idx)), index=idx, name="demand")
    return ci, demand


def test_expected_columns_present(hourly):
    feats = build_features(*hourly)
    for col in ["ci_lag_1", "ci_lag_2", "ci_lag_24", "ci_lag_48", "ci_lag_168",
                "demand_lag_168", "ci_roll6_mean", "ci_roll24_std", "demand_roll6_std",
                "hour_sin", "hour_cos", "day_of_week", "month", "day_of_year", "is_weekend"]:
        assert col in feats.columns, col


def test_lag_values_are_correct(hourly):
    ci, demand = hourly
    feats = build_features(ci, demand)
    t = 500
    assert feats["ci_lag_24"].iloc[t] == pytest.approx(ci.iloc[t - 24])
    assert feats["ci_lag_168"].iloc[t] == pytest.approx(ci.iloc[t - 168])


def test_no_leakage_feature_at_t_ignores_value_at_t(hourly):
    ci, demand = hourly
    base = build_features(ci, demand)
    perturbed_ci = ci.copy()
    t = 300
    perturbed_ci.iloc[t] += 1_000.0  # change CI(t) only
    pert = build_features(perturbed_ci, demand)
    # features at t must be unchanged: the target itself must not leak in
    cols = [c for c in base.columns if c.startswith("ci_")]
    assert np.allclose(base[cols].iloc[t].to_numpy(dtype=float),
                       pert[cols].iloc[t].to_numpy(dtype=float), equal_nan=True)


def test_rolling_window_ends_at_t_minus_1(hourly):
    ci, demand = hourly
    feats = build_features(ci, demand)
    t = 400
    expected = ci.iloc[t - 6:t].mean()  # 6 hours ending at t-1
    assert feats["ci_roll6_mean"].iloc[t] == pytest.approx(expected)


def test_is_weekend_matches_local_day(hourly):
    ci, demand = hourly
    feats = build_features(ci, demand)
    local = ci.index.tz_convert("America/Los_Angeles")
    expected = (np.asarray(local.day_of_week) >= 5).astype(int)
    assert (expected == feats["is_weekend"].to_numpy()).all()


def test_naive_index_rejected():
    idx = pd.date_range("2023-01-01", periods=5, freq="h")
    s = pd.Series(range(5), index=idx, dtype=float)
    with pytest.raises(ValueError):
        build_features(s, s)
