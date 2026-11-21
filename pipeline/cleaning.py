"""Cleaning rules: hourly grid, sparse-day drop, imputation flags, outlier flags, time columns."""

from datetime import timedelta

import numpy as np
import pandas as pd


def to_hourly_utc_grid(series, start, end):
    """Reindex to a complete hourly UTC grid so every missing hour is explicit."""
    grid = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    return series.reindex(grid)


def drop_sparse_days(series, max_missing_frac=0.10):
    """Drop whole UTC days where more than max_missing_frac of hours are NaN.

    Returns (cleaned_series, dropped_days) where dropped_days is a list of dates.
    Dropped hours are removed from the index, not filled.
    """
    missing = series.isna()
    day = series.index.normalize()
    frac_missing = missing.groupby(day).mean()
    dropped_days = frac_missing.index[frac_missing > max_missing_frac]
    keep = ~day.isin(dropped_days)
    return series[keep], list(dropped_days.date)


def impute_short_gaps(series, limit=3):
    """Linearly interpolate gaps up to `limit` hours long.

    Returns (filled_series, imputed_flag). imputed_flag is True where a value
    was filled. Longer gaps stay NaN so they are visible downstream.
    """
    filled = series.interpolate(method="linear", limit=limit, limit_area="inside")
    imputed_flag = series.isna() & filled.notna()
    return filled, imputed_flag


def flag_outliers(gen, nameplate_mw):
    """Flag values without changing them.

    gen : Series of hourly generation (MW).
    nameplate_mw : float, or Series aligned to gen, giving capacity per fuel.
    Returns a DataFrame of boolean flags: above_nameplate, negative.
    """
    return pd.DataFrame(
        {
            "above_nameplate": gen > nameplate_mw,
            "negative": gen < 0,
        },
        index=gen.index,
    )


def add_time_columns(index_utc, local_tz="America/Los_Angeles"):
    """Return local time and a DST flag for a UTC index.

    Storage stays in UTC. The local column is for seasonality and
    demand patterns only. dst_flag is True when the local offset is daylight time.
    """
    if index_utc.tz is None:
        raise ValueError("index must be timezone-aware UTC")
    local = index_utc.tz_convert(local_tz)
    dst_flag = [t.dst() != timedelta(0) for t in local.to_pydatetime()]
    return pd.DataFrame(
        {
            "local_time": local,
            "dst_flag": np.array(dst_flag, dtype=bool),
        },
        index=index_utc,
    )
