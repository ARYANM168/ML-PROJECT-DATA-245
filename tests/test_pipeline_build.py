import numpy as np
import pandas as pd
import pytest

from pipeline.features import build_features_lead

from pipeline.checks import no_future_check
from pipeline.features import build_hourly_weather_features, build_weather_features
from pipeline.weather import fill_daily_climatology


def test_daily_climatology_fills_gap_from_reference_years_only():
    idx = pd.date_range("2022-06-01", "2024-06-30", freq="D")
    s = pd.Series(np.nan, index=idx)
    s[idx < "2024-01-01"] = 20.0
    s["2024-06-10"] = np.nan  # gap in the post-reference period
    s["2024-03-01"] = 1000.0  # test-period value must not affect the reference climatology
    filled, flag = fill_daily_climatology(s, reference_end="2024-01-01")
    assert flag.loc["2024-06-10"]
    assert filled.loc["2024-06-10"] == pytest.approx(20.0)


def test_weather_features_use_previous_local_day():
    idx = pd.date_range("2024-03-10 00:00", "2024-03-11 23:00", freq="h", tz="UTC")
    daily = pd.DataFrame({"TMAX": [10.0, 20.0, 30.0, 40.0]},
                         index=pd.date_range("2024-03-09", periods=4, freq="D"))
    wx = build_weather_features(daily, idx)
    # local day 2024-03-10 uses 2024-03-09's value for every hour of that day
    local_day10 = idx.tz_convert("America/Los_Angeles").tz_localize(None).normalize() == pd.Timestamp("2024-03-10")
    assert (wx.loc[local_day10, "wx_prev_tmax"] == 10.0).all()


def test_no_future_check_passes_and_control_detects_on_synthetic_series():
    idx = pd.date_range("2023-01-01", periods=24 * 60, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    ci = pd.Series(rng.normal(300, 20, len(idx)), index=idx)
    dem = pd.Series(rng.normal(20000, 1000, len(idx)), index=idx)
    wx = pd.DataFrame({"temp": rng.normal(10, 3, len(idx)), "wind": rng.normal(3, 1, len(idx))}, index=idx)
    res = no_future_check(ci, dem, wx, lead=1, use_weather=True, n_samples=10)
    assert res["passed_future_perturbation"] == res["samples"]
    assert res["positive_control_detected"] == res["samples"]
    assert res["result"] == "PASS"


def test_lead_features_ignore_the_most_recent_week():
    idx = pd.date_range("2023-01-01", periods=24 * 900, freq="h", tz="UTC")
    rng = np.random.default_rng(2)
    ci = pd.Series(rng.normal(300, 20, len(idx)), index=idx)
    dem = pd.Series(rng.normal(20000, 1000, len(idx)), index=idx)
    base = build_features_lead(ci, dem, lead=168)
    t = 2000
    ci2 = ci.copy(); ci2.iloc[t - 167:t] += 1e4  # the 167 hours before t are inside the lead window
    dem2 = dem.copy(); dem2.iloc[t - 167:t] += 1e4
    shifted = build_features_lead(ci2, dem2, lead=168)
    cols = [c for c in base.columns if c.startswith(("ci_", "demand_"))]
    assert np.allclose(base[cols].iloc[t].to_numpy(dtype=float), shifted[cols].iloc[t].to_numpy(dtype=float), equal_nan=True)
    assert base.shape[1] == 26
