"""Feature engineering. Every feature at time t uses only data up to t-1."""

import numpy as np
import pandas as pd

LAGS = (1, 2, 24, 48, 168)
ROLL_WINDOWS = (6, 24)
LOCAL_TZ = "America/Los_Angeles"


def build_features(ci, demand, local_tz=LOCAL_TZ):
    """Build the feature table for predicting CI(t).

    ci, demand : Series on a UTC hourly index. CI(t) is the target and is
    not included as a feature at time t.
    Returns a DataFrame with one row per hour (same index).
    """
    if ci.index.tz is None:
        raise ValueError("index must be timezone-aware UTC")
    out = {}

    for name, s in (("ci", ci), ("demand", demand)):
        for k in LAGS:
            out[f"{name}_lag_{k}"] = s.shift(k)
        past = s.shift(1)  # window ends at t-1, so CI(t) never enters its own features
        for w in ROLL_WINDOWS:
            roll = past.rolling(w, min_periods=w)
            out[f"{name}_roll{w}_mean"] = roll.mean()
            out[f"{name}_roll{w}_std"] = roll.std()

    local = ci.index.tz_convert(local_tz)
    hour = np.asarray(local.hour)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = np.asarray(local.day_of_week)
    out["day_of_week"] = dow
    out["month"] = np.asarray(local.month)
    out["day_of_year"] = np.asarray(local.day_of_year)
    out["is_weekend"] = (dow >= 5).astype(int)

    return pd.DataFrame(out, index=ci.index)


def build_weather_features(daily, index_utc, local_tz=LOCAL_TZ):
    """Attach the PREVIOUS local day's weather to every hour.

    daily : DataFrame indexed by naive local calendar dates (midnight).
    Daily maxima and minima are not known until the day ends, so the same
    day's values would leak future information into earlier hours. Using
    yesterday's values keeps every feature at t based on data before t.
    """
    if index_utc.tz is None:
        raise ValueError("index must be timezone-aware UTC")
    local_day = pd.DatetimeIndex(index_utc.tz_convert(local_tz).tz_localize(None).normalize())
    prev_day = local_day - pd.Timedelta(days=1)
    out = daily.reindex(prev_day)
    out.index = index_utc
    out.columns = [f"wx_prev_{c.lower()}" for c in daily.columns]
    return out


def build_features_lead(ci, demand, lead=168, local_tz=LOCAL_TZ):
    """Features for a fixed-lead forecast: every input at time t is at least `lead` hours old.

    A forecast made at origin o = t - lead can use only data up to o, so no
    shift is smaller than `lead`. No weather is included: a 7-day forecast would
    need forecast weather, which this project does not have.
    """
    if ci.index.tz is None:
        raise ValueError("index must be timezone-aware UTC")
    out = {}
    for name, s in (("ci", ci), ("demand", demand)):
        lags = {k: s.shift(k * lead) for k in (1, 2, 3, 4)}
        for k, series in lags.items():
            out[f"{name}_lag_{k}w"] = series
        out[f"{name}_profile_4w_mean"] = pd.concat(lags.values(), axis=1).mean(axis=1, skipna=False)
        past = s.shift(lead)
        r24 = past.rolling(24, min_periods=24)
        r168 = past.rolling(168, min_periods=168)
        r672 = past.rolling(672, min_periods=672)
        out[f"{name}_roll24_mean"] = r24.mean()
        out[f"{name}_roll24_std"] = r24.std()
        out[f"{name}_roll168_mean"] = r168.mean()
        out[f"{name}_roll168_std"] = r168.std()
        out[f"{name}_roll672_mean"] = r672.mean()

    local = ci.index.tz_convert(local_tz)
    hour = np.asarray(local.hour)
    dow = np.asarray(local.day_of_week)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week"] = dow
    out["month"] = np.asarray(local.month)
    out["day_of_year"] = np.asarray(local.day_of_year)
    out["is_weekend"] = (dow >= 5).astype(int)
    return pd.DataFrame(out, index=ci.index)


def build_hourly_weather_features(temp, wind):
    """Hourly weather known before t: the observation at t-1 and the 24-hour mean ending at t-1.

    temp, wind : Series on the same UTC hourly index (already gap-filled).
    """
    out = {
        "wx_temp_lag1": temp.shift(1),
        "wx_temp_roll24_mean": temp.shift(1).rolling(24, min_periods=24).mean(),
        "wx_wind_lag1": wind.shift(1),
        "wx_wind_roll24_mean": wind.shift(1).rolling(24, min_periods=24).mean(),
    }
    return pd.DataFrame(out, index=temp.index)
