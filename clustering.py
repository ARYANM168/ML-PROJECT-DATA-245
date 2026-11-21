"""
clustering.py - discover day types from 24-hour carbon-intensity profiles.

Two methods are compared on the training days only:
  * k-means (Euclidean, many random restarts)
  * agglomerative hierarchical clustering with Ward linkage

The number of day types k is chosen from silhouette, Davies-Bouldin and
Calinski-Harabasz scores.  Test days are then assigned to the nearest
training centroid, so the day-type definitions never see test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

import config


def shape_normalise(X: np.ndarray) -> np.ndarray:
    """Remove each day's mean and scale by its std: keeps only the daily shape."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def scan_k(X: np.ndarray, k_values=config.K_RANGE) -> pd.DataFrame:
    """Score k-means and Ward solutions for every candidate k."""
    Z = linkage(X, method="ward")
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=config.N_INIT, random_state=config.RANDOM_STATE).fit(X)
        ward = fcluster(Z, k, criterion="maxclust") - 1
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "kmeans_silhouette": silhouette_score(X, km.labels_),
                "kmeans_davies_bouldin": davies_bouldin_score(X, km.labels_),
                "kmeans_calinski": calinski_harabasz_score(X, km.labels_),
                "ward_silhouette": silhouette_score(X, ward),
                "ward_davies_bouldin": davies_bouldin_score(X, ward),
                "kmeans_vs_ward_ARI": adjusted_rand_score(km.labels_, ward),
            }
        )
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame) -> int:
    """Pick k by majority vote of three internal indices, restricted to k >= 3.

    k = 2 almost always wins on silhouette for this data (it just splits
    'solar-rich' from 'solar-poor' days), which is too coarse to be useful for
    scheduling, so the search starts at 3.  Each index ranks the candidates and
    the k with the best average rank is chosen.
    """
    s = scores[scores["k"] >= 3].copy()
    s["r_sil"] = s["kmeans_silhouette"].rank(ascending=False)
    s["r_db"] = s["kmeans_davies_bouldin"].rank(ascending=True)
    s["r_ch"] = s["kmeans_calinski"].rank(ascending=False)
    s["mean_rank"] = s[["r_sil", "r_db", "r_ch"]].mean(axis=1)
    return int(s.sort_values(["mean_rank", "k"]).iloc[0]["k"])


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=config.N_INIT, random_state=config.RANDOM_STATE).fit(X)


def order_clusters_by_mean(centroids: np.ndarray) -> np.ndarray:
    """Return a relabelling map so cluster 0 = cleanest day type."""
    order = np.argsort(centroids.mean(axis=1))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap


def describe_clusters(days: pd.DataFrame, labels: pd.Series, calendar: pd.DataFrame) -> pd.DataFrame:
    """One summary row per day type (used for naming and for the report table)."""
    df = calendar.join(labels.rename("cluster"))
    prof = days.join(labels.rename("cluster")).groupby("cluster").mean()
    out = pd.DataFrame(index=sorted(labels.unique()))
    out["n_days"] = df.groupby("cluster").size()
    out["share_pct"] = 100 * out["n_days"] / out["n_days"].sum()
    out["mean_ci"] = prof.mean(axis=1)
    out["min_ci"] = prof.min(axis=1)
    out["max_ci"] = prof.max(axis=1)
    out["cleanest_hour"] = prof.values.argmin(axis=1)
    out["dirtiest_hour"] = prof.values.argmax(axis=1)
    out["weekend_pct"] = 100 * df.groupby("cluster")["is_weekend"].mean()
    out["top_months"] = df.groupby("cluster")["month"].agg(
        lambda m: ",".join(str(x) for x in m.value_counts().index[:3])
    )
    out["dominant_season"] = df.groupby("cluster")["season"].agg(lambda s: s.value_counts().index[0])
    return out


def name_clusters(desc: pd.DataFrame) -> dict[int, str]:
    """Readable names from the daily level and the dominant season."""
    names = {}
    for c, row in desc.iterrows():
        level = "Low" if row["mean_ci"] < desc["mean_ci"].quantile(0.34) else (
            "High" if row["mean_ci"] > desc["mean_ci"].quantile(0.66) else "Mid")
        names[c] = f"C{c}: {level}-carbon day (mostly {row['dominant_season'].lower()})"
    return names
