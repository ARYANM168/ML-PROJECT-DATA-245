"""Weather helpers: area-weighted station averages and climatological gap-fill."""

import numpy as np
import pandas as pd


def area_weighted_mean(values_wide, weights):
    """Average station values to one series per balancing authority.

    values_wide : DataFrame, index = UTC hourly timestamps, columns = stations.
    weights : DataFrame, index = stations, columns = balancing authorities.
        Each column holds the area or population share of each station in that BA.
        Column weights are normalized over stations with a non-missing value at each hour.
    Returns DataFrame, index = timestamps, columns = BAs.
    """
    out = {}
    for ba in weights.columns:
        w = weights[ba].reindex(values_wide.columns).fillna(0.0)
        present = values_wide.notna().mul(w, axis=1)
        num = values_wide.fillna(0.0).mul(w, axis=1).sum(axis=1)
        den = present.sum(axis=1)
        out[ba] = (num / den).where(den > 0)
    return pd.DataFrame(out, index=values_wide.index)


def fill_with_climatology(series, reference_end):
    """Fill NaN hours with the mean of the same (month, hour) from reference years.

    Only observations up to `reference_end` (exclusive) are used, so the
    climatology does not use test-period data.
    Returns (filled_series, filled_flag).
    """
    ref = series[series.index < pd.Timestamp(reference_end, tz="UTC")].dropna()
    local_ref = ref.index.tz_convert("America/Los_Angeles")
    clim = ref.groupby([local_ref.month, local_ref.hour]).mean()

    local_all = series.index.tz_convert("America/Los_Angeles")
    keys = list(zip(local_all.month, local_all.hour))
    fill_values = pd.Series([clim.get(k, np.nan) for k in keys], index=series.index)
    filled_flag = series.isna() & fill_values.notna()
    return series.where(~filled_flag, fill_values), filled_flag


def fill_daily_climatology(series, reference_end):
    """Fill NaN days with the mean of the same day-of-year in years before reference_end.

    series : daily Series with a naive DatetimeIndex (local calendar dates).
    Returns (filled_series, filled_flag).
    """
    ref = series[series.index < pd.Timestamp(reference_end)].dropna()
    clim = ref.groupby(np.asarray(ref.index.day_of_year)).mean()
    doy = np.asarray(series.index.day_of_year)
    fill = pd.Series(clim.reindex(doy).to_numpy(), index=series.index)
    filled_flag = series.isna() & fill.notna()
    return series.where(~filled_flag, fill), filled_flag
