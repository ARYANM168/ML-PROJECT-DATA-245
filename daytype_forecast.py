"""
daytype_forecast.py - predict tomorrow's day type before it happens.

A scheduler has to commit to tomorrow's charging window the evening before, so
the day type must be predicted from information available at that time:
  * calendar of the target day (month, day of week, weekend flag)
  * the full 24-hour profile of the day before
  * the day type of the day before

A random-forest classifier is trained on the training period only.  Its
accuracy is compared with "persistence" (tomorrow = same type as today) and
"climatology" (most common type for the calendar month).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config


def build_features(days: pd.DataFrame, labels: pd.Series, k: int) -> pd.DataFrame:
    """Features for target date d use only data dated d-1 or the calendar of d."""
    idx = pd.DatetimeIndex(days.index)
    prev_day = idx - pd.Timedelta(days=1)
    prev_profile = days.reindex(prev_day)
    prev_profile.index = idx
    prev_profile.columns = [f"prev_{c}" for c in days.columns]
    prev_label = labels.reindex(prev_day)
    prev_label.index = idx

    X = pd.DataFrame(index=idx)
    X["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    X["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    X["dayofweek"] = idx.dayofweek
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    X = X.join(prev_profile)
    X["prev_mean"] = prev_profile.mean(axis=1)
    X["prev_range"] = prev_profile.max(axis=1) - prev_profile.min(axis=1)
    for c in range(k):
        X[f"prev_is_c{c}"] = (prev_label == c).astype(float).where(prev_label.notna())
    X["prev_label"] = prev_label
    return X


def train_and_predict(days, labels, k, train_index, test_index):
    X = build_features(days, labels, k)
    y = labels.reindex(X.index)
    feat_cols = [c for c in X.columns if c != "prev_label"]

    tr = X.loc[X.index.isin(train_index)].dropna()
    te = X.loc[X.index.isin(test_index)].dropna()
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, random_state=config.RANDOM_STATE, n_jobs=-1
    ).fit(tr[feat_cols], y.loc[tr.index])
    pred = pd.Series(clf.predict(te[feat_cols]), index=te.index, name="pred_cluster")

    # Baselines
    persistence = te["prev_label"].astype(int)
    month_mode = (
        pd.DataFrame({"m": tr.index.month, "y": y.loc[tr.index].values})
        .groupby("m")["y"].agg(lambda s: s.value_counts().index[0])
    )
    climatology = pd.Series(te.index.month.map(month_mode), index=te.index)

    truth = y.loc[te.index].astype(int)
    metrics = pd.DataFrame(
        {
            "accuracy": [
                accuracy_score(truth, pred),
                accuracy_score(truth, persistence),
                accuracy_score(truth, climatology),
            ],
            "macro_f1": [
                f1_score(truth, pred, average="macro"),
                f1_score(truth, persistence, average="macro"),
                f1_score(truth, climatology, average="macro"),
            ],
        },
        index=["random_forest", "persistence", "monthly_climatology"],
    )
    cm = pd.DataFrame(confusion_matrix(truth, pred, labels=range(k)),
                      index=[f"true_C{i}" for i in range(k)],
                      columns=[f"pred_C{i}" for i in range(k)])
    importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    return pred, metrics, cm, importances
