import numpy as np
import pandas as pd
import pytest

from pipeline.cleaning import (
    add_time_columns,
    drop_sparse_days,
    flag_outliers,
    impute_short_gaps,
    to_hourly_utc_grid,
)


def test_grid_makes_missing_hours_explicit():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"], utc=True))
    grid = to_hourly_utc_grid(s, "2024-01-01 00:00", "2024-01-01 03:00")
    assert len(grid) == 4
    assert grid.isna().sum() == 2


def test_drop_day_over_ten_percent_missing():
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    vals = np.ones(48)
    vals[:3] = np.nan  # day 1: 3/24 = 12.5% missing -> drop
    vals[30:32] = np.nan  # day 2: 2/24 = 8.3% missing -> keep
    s = pd.Series(vals, index=idx)
    cleaned, dropped = drop_sparse_days(s, max_missing_frac=0.10)
    assert dropped == [pd.Timestamp("2024-01-01").date()]
    assert cleaned.index.normalize().nunique() == 1
    assert len(cleaned) == 24


def test_impute_short_gap_and_flag():
    s = pd.Series([1.0, np.nan, np.nan, 4.0], index=pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"))
    filled, flag = impute_short_gaps(s, limit=3)
    assert filled.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert flag.tolist() == [False, True, True, False]


def test_long_gap_not_imputed():
    s = pd.Series([1.0] + [np.nan] * 5 + [7.0], index=pd.date_range("2024-01-01", periods=7, freq="h", tz="UTC"))
    filled, flag = impute_short_gaps(s, limit=3)
    assert flag.sum() == 3  # first 3 NaNs filled
    assert filled.isna().sum() == 2  # last 2 NaNs stay missing
    assert not flag.iloc[4]  # beyond the 3-hour limit


def test_outliers_flagged_not_deleted():
    gen = pd.Series([50.0, -5.0, 150.0], index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    flags = flag_outliers(gen, nameplate_mw=100.0)
    assert len(flags) == 3
    assert flags["negative"].tolist() == [False, True, False]
    assert flags["above_nameplate"].tolist() == [False, False, True]
    assert gen.tolist() == [50.0, -5.0, 150.0]


def test_dst_flag_and_local_time_for_spring_forward():
    # 2024-03-10 is the US spring-forward date in America/Los_Angeles
    idx = pd.date_range("2024-03-09 12:00", "2024-03-11 12:00", freq="h", tz="UTC")
    out = add_time_columns(idx, local_tz="America/Los_Angeles")
    assert not out["dst_flag"].iloc[0]  # Mar 9 local time is PST
    assert out["dst_flag"].iloc[-1]  # Mar 11 local time is PDT
    assert str(out["local_time"].dt.tz) == "America/Los_Angeles"


def test_naive_index_rejected():
    with pytest.raises(ValueError):
        add_time_columns(pd.date_range("2024-01-01", periods=2, freq="h"))
