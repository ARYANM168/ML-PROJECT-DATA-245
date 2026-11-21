"""
data_prep.py - turn the hourly carbon-intensity series into a day x hour matrix.

Each row of the matrix is one calendar day (local time, America/Los_Angeles)
and each of the 24 columns is the carbon intensity in that hour.  Days that are
still incomplete after filling short gaps (including the 23-hour spring DST
day) are dropped, so every profile has exactly 24 valid values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def load_hourly(path=config.DATA_PATH) -> pd.Series:
    """Load the cached CAISO series onto an exact hourly grid."""
    df = pd.read_csv(path, parse_dates=[config.TIME_COL]).sort_values(config.TIME_COL)
    s = df.set_index(config.TIME_COL)[config.TARGET_COL].astype(float)
    s = s[~s.index.duplicated(keep="first")].asfreq("h")
    s = s.loc[s.first_valid_index():]
    # Zero or negative intensity in a fossil-heavy system is a reporting error.
    s = s.where(s >= config.MIN_VALID_CI)
    s = s.interpolate(limit=config.MAX_INTERP_HOURS, limit_area="inside")
    s.name = config.TARGET_COL
    return s


def daily_matrix(s: pd.Series) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns h00 ... h23."""
    frame = pd.DataFrame({"ci": s, "date": s.index.normalize(), "hour": s.index.hour})
    wide = frame.pivot_table(index="date", columns="hour", values="ci", aggfunc="first")
    wide = wide.reindex(columns=range(24))
    wide = wide.dropna(how="any")
    wide.columns = [f"h{h:02d}" for h in range(24)]
    wide.index.name = "date"
    return wide


def add_calendar(days: pd.DataFrame) -> pd.DataFrame:
    """Calendar descriptors used to interpret and predict day types."""
    out = pd.DataFrame(index=days.index)
    idx = pd.DatetimeIndex(days.index)
    out["year"] = idx.year
    out["month"] = idx.month
    out["dayofweek"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["season"] = idx.month.map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
    )
    out["daily_mean"] = days.mean(axis=1)
    out["daily_min"] = days.min(axis=1)
    out["daily_max"] = days.max(axis=1)
    out["daily_range"] = out["daily_max"] - out["daily_min"]
    return out


def split_train_test(days: pd.DataFrame):
    """Chronological split - no shuffling, no test days used for fitting."""
    train = days.loc[: config.TRAIN_END]
    test = days.loc[config.TEST_START:]
    return train, test


if __name__ == "__main__":
    series = load_hourly()
    days = daily_matrix(series)
    train, test = split_train_test(days)
    print(f"Hourly span: {series.index.min()} -> {series.index.max()}")
    print(f"Complete days: {len(days)}  (train {len(train)}, test {len(test)})")
    print(days.describe().T[["mean", "std", "min", "max"]].round(1))
