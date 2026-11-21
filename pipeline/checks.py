"""No-future-data check: perturb everything at or after t and confirm features at t do not move.

A positive control runs alongside it: perturbing the newest input the features at t
are allowed to use (t - lead) must change at least some features at t. Without the
control, a check that never detects anything would pass silently.
"""

import numpy as np
import pandas as pd

from .features import build_features, build_features_lead, build_hourly_weather_features

PERTURB = 1e4


def _feature_table(ci, demand, wx, lead, use_weather):
    feats = build_features(ci, demand) if lead == 1 else build_features_lead(ci, demand, lead)
    if not use_weather:
        return feats
    wx_feats = build_hourly_weather_features(wx["temp"], wx["wind"])
    return pd.concat([feats, wx_feats], axis=1)


def no_future_check(ci, demand, wx, lead=1, use_weather=True, n_samples=40, seed=0):
    """Return a dict of check results.

    ci, demand : hourly Series on a UTC index (same index).
    wx : hourly DataFrame with columns temp and wind on the same UTC index (used only if use_weather).
    lead : forecast lead in hours (1 for the 1-hour model, 168 for the 7-day model).
    """
    rng = np.random.default_rng(seed)
    base = _feature_table(ci, demand, wx, lead, use_weather)
    warmup = 4 * lead + 700  # enough history for the longest lag and rolling window
    candidates = np.arange(warmup, len(ci))
    picks = rng.choice(candidates, size=min(n_samples, len(candidates)), replace=False)

    passed, control_detected, failures = 0, 0, []
    for t in picks:
        # 1. everything at or after t moves: features at t must not change
        ci2, dem2, wx2 = ci.copy(), demand.copy(), wx.copy()
        ci2.iloc[t:] = ci2.iloc[t:] + PERTURB
        dem2.iloc[t:] = dem2.iloc[t:] + PERTURB
        wx2.iloc[t:] = wx2.iloc[t:] + PERTURB
        after = _feature_table(ci2, dem2, wx2, lead, use_weather).iloc[t]
        same = np.allclose(base.iloc[t].to_numpy(dtype=float), after.to_numpy(dtype=float), equal_nan=True)
        if same:
            passed += 1
        else:
            failures.append(str(ci.index[t]))

        # 2. positive control: change only the newest allowed input (t - lead); features at t should move
        ci3, dem3 = ci.copy(), demand.copy()
        ci3.iloc[t - lead] = ci3.iloc[t - lead] + PERTURB
        dem3.iloc[t - lead] = dem3.iloc[t - lead] + PERTURB
        ctrl = _feature_table(ci3, dem3, wx, lead, use_weather).iloc[t]
        moved = not np.allclose(base.iloc[t].to_numpy(dtype=float), ctrl.to_numpy(dtype=float), equal_nan=True)
        control_detected += int(moved)

    n = len(picks)
    return {
        "lead_hours": int(lead),
        "samples": int(n),
        "features_checked": int(base.shape[1]),
        "passed_future_perturbation": int(passed),
        "failed_timestamps": failures,
        "positive_control_detected": int(control_detected),
        "result": "PASS" if passed == n and control_detected == n else "FAIL",
    }
