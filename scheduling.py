"""
scheduling.py - simulate the carbon saved by shifting flexible load.

Scenario: every day a flexible job (e.g. EV fleet charging, a batch compute
job, water pumping) needs FLEX_ENERGY_MWH of energy delivered as one
contiguous block of W hours somewhere inside the calendar day.  The energy is
spread evenly across the W hours.  Emissions are evaluated against the ACTUAL
hourly intensity of the test day:

    emissions_kg = (E / W) * sum_{t=s}^{s+W-1} CI_actual[t]

(1 MWh x 1 gCO2/kWh = 1 kg CO2, so kg per MWh equals g per kWh numerically.)

Strategies differ only in the information used to choose the start hour s:
  flat             - no block; load spread over all 24 h (daily mean)
  evening_bau      - fixed 18:00 block (typical after-work EV plug-in)
  static_tou       - best block of the training-period average day
  monthly_static   - best block of the training average for that month
  yesterday        - best block of yesterday's actual profile (seasonal naive)
  cluster_pred     - best block of the predicted day type's centroid
  cluster_known    - best block of the true day type's centroid (upper bound
                     for a fixed-centroid day-type approach)
  recent_7d        - best block of the average of the last 7 observed days
  cluster_adaptive - best block of the average of the 14 most recent days that
                     belong to the predicted day type (day-type analogue
                     method; adapts to drift in the grid mix)
  oracle           - best block of the actual profile (perfect foresight)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def block_sums(profile: np.ndarray, W: int) -> np.ndarray:
    """Sum of every contiguous W-hour block inside one 24-h profile."""
    c = np.concatenate([[0.0], np.cumsum(profile)])
    return c[W:] - c[:-W]          # length 25 - W, start hours 0 .. 24-W


def best_start(profile: np.ndarray, W: int) -> int:
    return int(np.argmin(block_sums(profile, W)))


def emissions_for_start(actual: np.ndarray, start: int, W: int, energy=config.FLEX_ENERGY_MWH) -> float:
    return energy / W * actual[start : start + W].sum()


ADAPTIVE_LOOKBACK_DAYS = 120   # history searched for same-type analogue days
ADAPTIVE_N_ANALOGUES = 14
RECENT_DAYS = 7


def choose_starts(test_days, train_days, centroids, pred_cluster, true_cluster, W,
                  all_labels=None):
    """Return a DataFrame of start hours (one column per strategy)."""
    starts = pd.DataFrame(index=test_days.index)
    starts["evening_bau"] = min(config.EVENING_START, 24 - W)
    starts["static_tou"] = best_start(train_days.mean().to_numpy(), W)

    monthly = train_days.groupby(pd.DatetimeIndex(train_days.index).month).mean()
    month_best = {m: best_start(monthly.loc[m].to_numpy(), W) for m in monthly.index}
    starts["monthly_static"] = pd.DatetimeIndex(test_days.index).month.map(month_best)

    all_days = pd.concat([train_days, test_days]).sort_index()
    prev = all_days.reindex(pd.DatetimeIndex(test_days.index) - pd.Timedelta(days=1))
    prev.index = test_days.index
    yest = []
    for d, row in prev.iterrows():
        yest.append(best_start(row.to_numpy(), W) if row.notna().all() else starts.at[d, "monthly_static"])
    starts["yesterday"] = yest

    cent_best = {c: best_start(centroids[c], W) for c in range(len(centroids))}
    starts["cluster_pred"] = [
        cent_best[int(pred_cluster[d])] if d in pred_cluster.index else starts.at[d, "monthly_static"]
        for d in test_days.index
    ]
    starts["cluster_known"] = [cent_best[int(true_cluster[d])] for d in test_days.index]
    # adaptive strategies: only days strictly before d are used
    recent, adaptive = [], []
    pos = {d: i for i, d in enumerate(all_days.index)}
    for d in test_days.index:
        i = pos[d]
        hist = all_days.iloc[max(0, i - ADAPTIVE_LOOKBACK_DAYS): i]
        recent.append(best_start(hist.tail(RECENT_DAYS).mean().to_numpy(), W))
        if all_labels is not None and d in pred_cluster.index:
            same = hist[all_labels.reindex(hist.index).to_numpy() == int(pred_cluster[d])]
            same = same.tail(ADAPTIVE_N_ANALOGUES)
            adaptive.append(best_start(same.mean().to_numpy(), W) if len(same) else recent[-1])
        else:
            adaptive.append(recent[-1])
    starts["recent_7d"] = recent
    starts["cluster_adaptive"] = adaptive
    starts["oracle"] = [best_start(r, W) for r in test_days.to_numpy()]
    return starts.astype(int)


def simulate(test_days, train_days, centroids, pred_cluster, true_cluster, W=config.BLOCK_HOURS,
             all_labels=None):
    """Daily emissions (kg per FLEX_ENERGY_MWH) for every strategy."""
    starts = choose_starts(test_days, train_days, centroids, pred_cluster, true_cluster, W, all_labels)
    actual = test_days.to_numpy()
    out = pd.DataFrame(index=test_days.index)
    out["flat"] = config.FLEX_ENERGY_MWH * actual.mean(axis=1)
    for col in starts.columns:
        s = starts[col].to_numpy()
        out[col] = [emissions_for_start(actual[i], s[i], W) for i in range(len(s))]
    return out, starts


def summarise(daily: pd.DataFrame, baseline="evening_bau") -> pd.DataFrame:
    """Average intensity, savings vs. baselines and capture of the oracle gain."""
    mean = daily.mean()
    res = pd.DataFrame({"kg_per_MWh": mean})
    res["saving_vs_evening_kg"] = mean[baseline] - mean
    res["saving_vs_evening_pct"] = 100 * res["saving_vs_evening_kg"] / mean[baseline]
    res["saving_vs_flat_pct"] = 100 * (mean["flat"] - mean) / mean["flat"]
    gain_oracle = mean[baseline] - mean["oracle"]
    res["capture_of_oracle"] = res["saving_vs_evening_kg"] / gain_oracle
    # tonnes per year for a fleet shifting FLEET_FLEX_MWH_PER_DAY
    res["fleet_t_saved_per_year"] = (
        res["saving_vs_evening_kg"] * config.FLEET_FLEX_MWH_PER_DAY * 365 / 1000
    )
    order = ["evening_bau", "flat", "static_tou", "monthly_static", "yesterday",
             "cluster_pred", "cluster_known", "recent_7d", "cluster_adaptive", "oracle"]
    return res.loc[[o for o in order if o in res.index]]


def bootstrap_ci(daily: pd.DataFrame, a: str, b: str, n=2000, seed=config.RANDOM_STATE):
    """95% bootstrap interval of mean(a - b) over test days (paired)."""
    rng = np.random.default_rng(seed)
    diff = (daily[a] - daily[b]).to_numpy()
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
