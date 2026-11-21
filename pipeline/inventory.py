"""Raw-data inventory: missing hours, negatives, and values above nameplate, by BA and fuel.

Expected long-format input with columns:
    ba        : balancing authority code
    fuel      : EIA-930 fuel code
    timestamp : UTC hourly timestamp
    mw        : hourly generation (may be NaN)
    nameplate_mw : capacity for that BA and fuel (optional; NaN if unknown)
"""

import pandas as pd


def inventory(df, expected_hours_per_group=None):
    """Summarize data quality per (ba, fuel).

    expected_hours_per_group : int or None. If None, uses the span of
        timestamps per group on an hourly grid.
    Returns a DataFrame with one row per (ba, fuel).
    """
    rows = []
    for (ba, fuel), g in df.groupby(["ba", "fuel"]):
        ts = pd.DatetimeIndex(g["timestamp"])
        if expected_hours_per_group is not None:
            expected = expected_hours_per_group
        else:
            expected = int((ts.max() - ts.min()) / pd.Timedelta(hours=1)) + 1
        present = g["mw"].notna().sum()
        nameplate = g["nameplate_mw"]
        above = (g["mw"] > nameplate).sum() if nameplate.notna().any() else pd.NA
        rows.append(
            {
                "ba": ba,
                "fuel": fuel,
                "expected_hours": expected,
                "missing_hours": int(expected - present),
                "missing_share": float((expected - present) / expected) if expected else float("nan"),
                "negative_values": int((g["mw"] < 0).sum()),
                "above_nameplate": above,
            }
        )
    return pd.DataFrame(rows)
