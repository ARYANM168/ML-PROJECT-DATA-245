"""Train, validation, and test splits in UTC, plus rolling-origin folds.

Split boundaries follow the checklist: train 2019-2023, validate early 2024,
test late 2024-2025. The checklist does not say where Apr-Jun 2024 goes.
Validate is set to Jan-Jun 2024 and test to Jul 2024-Dec 2025 as a working
choice. Confirm with the team.
"""

import numpy as np
import pandas as pd

SPLITS = {
    "train": ("2019-01-01 00:00", "2023-12-31 23:00"),
    "validate": ("2024-01-01 00:00", "2024-06-30 23:00"),
    "test": ("2024-07-01 00:00", "2025-12-31 23:00"),
}
HORIZON_HOURS = 168  # 7 days ahead
ORIGIN_STEP_HOURS = 168  # new forecast origin every week


def _bounds(name):
    start, end = SPLITS[name]
    return pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")


def split_masks(index):
    """Return {split_name: boolean array} for a UTC DatetimeIndex. Splits do not overlap."""
    masks = {}
    for name in SPLITS:
        lo, hi = _bounds(name)
        masks[name] = np.asarray((index >= lo) & (index <= hi))
    return masks


def split_summary(index):
    """Record split dates and observation counts for the provenance record."""
    rows = []
    for name, mask in split_masks(index).items():
        lo, hi = _bounds(name)
        sel = index[mask]
        rows.append(
            {
                "split": name,
                "start_utc": lo.isoformat(),
                "end_utc": hi.isoformat(),
                "rows_in_range": int(mask.sum()),
                "first_observed": sel.min().isoformat() if len(sel) else None,
                "last_observed": sel.max().isoformat() if len(sel) else None,
            }
        )
    return pd.DataFrame(rows)


def rolling_origins(index, start, end):
    """Weekly forecast origins for 7-day-ahead rolling-origin validation.

    Each fold: train on rows strictly before `origin`, forecast the
    HORIZON_HOURS rows starting at `origin`. Folds whose forecast window
    would run past `end` are not created.
    """
    origin = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC")
    step = pd.Timedelta(hours=ORIGIN_STEP_HOURS)
    horizon = pd.Timedelta(hours=HORIZON_HOURS)
    folds = []
    while origin + horizon - pd.Timedelta(hours=1) <= end:
        forecast_end = origin + horizon - pd.Timedelta(hours=1)
        folds.append(
            {
                "origin": origin,
                "train_end": origin - pd.Timedelta(hours=1),
                "forecast_start": origin,
                "forecast_end": forecast_end,
                "n_train_rows": int((index < origin).sum()),
                "n_forecast_rows": int(((index >= origin) & (index <= forecast_end)).sum()),
            }
        )
        origin = origin + step
    return pd.DataFrame(folds)


def assert_no_future_in_train(train_index, origin):
    """Raise if any training timestamp is at or after the forecast origin."""
    origin = pd.Timestamp(origin)
    if len(train_index) and train_index.max() >= origin:
        raise AssertionError(
            f"future data in training set: max train time {train_index.max()} >= origin {origin}"
        )
