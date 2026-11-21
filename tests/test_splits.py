import pandas as pd
import pytest

from pipeline.splits import (
    assert_no_future_in_train,
    rolling_origins,
    split_masks,
    split_summary,
)


@pytest.fixture
def full_index():
    return pd.date_range("2019-01-01", "2025-12-31 23:00", freq="h", tz="UTC")


def test_splits_do_not_overlap(full_index):
    masks = split_masks(full_index)
    total = sum(m.sum() for m in masks.values())
    stacked = sum(m.astype(int) for m in masks.values())
    assert (stacked <= 1).all()
    assert total <= len(full_index)


def test_train_ends_before_validation_begins(full_index):
    masks = split_masks(full_index)
    assert full_index[masks["train"]].max() < full_index[masks["validate"]].min()
    assert full_index[masks["validate"]].max() < full_index[masks["test"]].min()


def test_split_summary_counts(full_index):
    summary = split_summary(full_index).set_index("split")
    assert summary.loc["train", "rows_in_range"] == 5 * 365 * 24 + 1 * 24  # 2019-2023 incl. 2020 leap day
    assert summary.loc["test", "first_observed"].startswith("2024-07-01")


def test_rolling_origins_are_weekly_and_seven_day_horizon(full_index):
    folds = rolling_origins(full_index, "2024-07-01", "2025-12-31 23:00")
    assert (folds["origin"].diff().dropna() == pd.Timedelta(days=7)).all()
    assert (folds["n_forecast_rows"] == 168).all()
    assert folds["forecast_end"].max() <= pd.Timestamp("2025-12-31 23:00", tz="UTC")


def test_rolling_train_rows_grow_and_precede_origin(full_index):
    folds = rolling_origins(full_index, "2024-07-01", "2024-08-31 23:00")
    assert (folds["n_train_rows"].diff().dropna() > 0).all()
    for origin, train_end in zip(folds["origin"], folds["train_end"]):
        assert train_end < origin


def test_future_in_training_is_caught(full_index):
    origin = pd.Timestamp("2024-07-01", tz="UTC")
    assert_no_future_in_train(full_index[full_index < origin], origin)  # clean case passes
    with pytest.raises(AssertionError):
        assert_no_future_in_train(full_index[full_index <= origin], origin)
