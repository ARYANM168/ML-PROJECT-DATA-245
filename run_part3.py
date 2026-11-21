"""
run_part3.py - end-to-end Part 3 pipeline.

    python run_part3.py

Steps
  1. load the hourly CAISO carbon-intensity series and build daily profiles
  2. chronological split (train <= 2023-12-31, test >= 2024-01-01)
  3. scan k for k-means and Ward clustering; choose k; fit final k-means
  4. label test days with the nearest training centroid
  5. train the day-ahead day-type classifier
  6. simulate the load-shifting strategies on the test days
  7. write tables (outputs/*.csv, outputs/part3_summary.txt) and figures
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

import clustering
import config
import data_prep
import daytype_forecast
import plots
import scheduling


def make_figures(scores, k, train, train_labels, names, Xtr, labels, summary, sens, daily, cm, test, starts):
    plots.k_selection(scores, k)
    plots.centroids(train, train_labels, names)
    plots.dendro(Xtr, k)
    plots.month_heatmap(labels, names)
    plots.year_mix(labels, names)
    plots.strategy_bars(summary)
    plots.block_sensitivity(sens)
    plots.yearly_saving(daily)
    plots.confusion(cm)
    plots.example_week(test, starts, config.BLOCK_HOURS, "2025-04-07")


def main():
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2 data
    series = data_prep.load_hourly()
    days = data_prep.daily_matrix(series)
    calendar = data_prep.add_calendar(days)
    train, test = data_prep.split_train_test(days)
    print(f"Complete days {len(days)}: train {len(train)} ({train.index.min().date()}..{train.index.max().date()}), "
          f"test {len(test)} ({test.index.min().date()}..{test.index.max().date()})")

    # 3 choose k on training days (raw gCO2/kWh profiles)
    Xtr = train.to_numpy()
    scores = clustering.scan_k(Xtr)
    scores.round(4).to_csv(config.OUTPUT_DIR / "k_selection_scores.csv", index=False)
    k = config.K_FINAL or clustering.choose_k(scores)
    print(scores.round(3).to_string(index=False))
    print(f"Chosen k = {k}")

    # shape-only ablation: does clustering on normalised shape agree?
    shape_km = clustering.fit_kmeans(clustering.shape_normalise(Xtr), k)

    km = clustering.fit_kmeans(Xtr, k)
    remap = clustering.order_clusters_by_mean(km.cluster_centers_)
    centroids = np.zeros_like(km.cluster_centers_)
    centroids[remap] = km.cluster_centers_
    train_labels = pd.Series(remap[km.labels_], index=train.index, name="cluster")
    test_labels = pd.Series(remap[km.predict(test.to_numpy())], index=test.index, name="cluster")
    labels = pd.concat([train_labels, test_labels])

    # Ward at the same k for agreement
    from scipy.cluster.hierarchy import fcluster, linkage
    ward = fcluster(linkage(Xtr, "ward"), k, criterion="maxclust")
    ari_ward = adjusted_rand_score(train_labels, ward)
    ari_shape = adjusted_rand_score(train_labels, shape_km.labels_)
    sil_test = silhouette_score(test.to_numpy(), test_labels)

    desc = clustering.describe_clusters(train, train_labels, calendar.loc[train.index])
    names = clustering.name_clusters(desc)
    desc["name"] = pd.Series(names)
    desc.round(2).to_csv(config.OUTPUT_DIR / "cluster_description.csv")
    pd.DataFrame(centroids, columns=days.columns,
                 index=[f"C{i}" for i in range(k)]).round(2).to_csv(config.OUTPUT_DIR / "cluster_centroids.csv")
    labels.to_frame().join(calendar[["month", "dayofweek", "season", "daily_mean"]]).to_csv(
        config.OUTPUT_DIR / "day_labels.csv")
    test_share = test_labels.value_counts(normalize=True).sort_index() * 100
    print(desc.round(1).to_string())

    # 5 day-ahead classifier
    pred, clf_metrics, cm, importances = daytype_forecast.train_and_predict(
        days, labels, k, train.index, test.index)
    clf_metrics.round(4).to_csv(config.OUTPUT_DIR / "daytype_classifier_metrics.csv")
    cm.to_csv(config.OUTPUT_DIR / "daytype_confusion_matrix.csv")
    importances.head(15).round(4).to_csv(config.OUTPUT_DIR / "daytype_feature_importance.csv")
    print(clf_metrics.round(3))

    # 6 scheduling simulation
    daily, starts = scheduling.simulate(test, train, centroids, pred, test_labels, config.BLOCK_HOURS, labels)
    daily.round(3).to_csv(config.OUTPUT_DIR / "schedule_daily_emissions.csv")
    starts.to_csv(config.OUTPUT_DIR / "schedule_start_hours.csv")
    summary = scheduling.summarise(daily)
    summary.round(3).to_csv(config.OUTPUT_DIR / "schedule_summary.csv")
    print(summary.round(2).to_string())

    clean = scheduling.summarise(daily.loc[: config.CLEAN_TEST_END])
    clean.round(3).to_csv(config.OUTPUT_DIR / "schedule_summary_clean_2024H1.csv")
    print("Clean pre-category-change window (2024-01..2024-06):")
    print(clean.round(2).to_string())

    boot = {s: scheduling.bootstrap_ci(daily, "evening_bau", s)
            for s in ["static_tou", "monthly_static", "yesterday", "cluster_pred", "recent_7d",
                      "cluster_adaptive", "oracle"]}
    boot["cluster_pred_vs_monthly_static"] = scheduling.bootstrap_ci(daily, "monthly_static", "cluster_pred")
    boot["cluster_pred_vs_yesterday"] = scheduling.bootstrap_ci(daily, "yesterday", "cluster_pred")
    boot["cluster_adaptive_vs_static_tou"] = scheduling.bootstrap_ci(daily, "static_tou", "cluster_adaptive")
    boot["cluster_adaptive_vs_recent_7d"] = scheduling.bootstrap_ci(daily, "recent_7d", "cluster_adaptive")

    # where does the true clean window sit, year by year?  (explains drift)
    oracle_start = starts["oracle"].groupby(pd.DatetimeIndex(starts.index).year).agg(
        lambda x: x.value_counts().index[0])
    train_oracle = pd.Series([scheduling.best_start(r, config.BLOCK_HOURS) for r in Xtr], index=train.index)
    drift = pd.concat([train_oracle.groupby(train_oracle.index.year).agg(lambda x: x.value_counts().index[0]),
                       oracle_start]).rename("most_common_best_start_hour")
    drift.to_csv(config.OUTPUT_DIR / "best_window_drift_by_year.csv")

    by_cluster = daily.join(test_labels).groupby("cluster").mean()
    by_cluster["saving_pred_pct"] = 100 * (by_cluster.evening_bau - by_cluster.cluster_pred) / by_cluster.evening_bau
    by_cluster["saving_oracle_pct"] = 100 * (by_cluster.evening_bau - by_cluster.oracle) / by_cluster.evening_bau
    by_cluster["saving_adaptive_pct"] = 100 * (by_cluster.evening_bau - by_cluster.cluster_adaptive) / by_cluster.evening_bau
    by_cluster["oracle_minus_bau_kg"] = by_cluster.evening_bau - by_cluster.oracle
    by_cluster["n_days"] = test_labels.value_counts().sort_index()
    by_cluster.round(2).to_csv(config.OUTPUT_DIR / "schedule_by_cluster.csv")

    by_year = daily.groupby(pd.DatetimeIndex(daily.index).year).mean()
    by_year.round(2).to_csv(config.OUTPUT_DIR / "schedule_by_year.csv")

    sens_rows = {}
    for W in config.BLOCK_SENSITIVITY:
        d_w, _ = scheduling.simulate(test, train, centroids, pred, test_labels, W, labels)
        sens_rows[W] = scheduling.summarise(d_w)["saving_vs_evening_pct"]
    sens = pd.DataFrame(sens_rows).T
    sens.index.name = "block_hours"
    sens.round(2).to_csv(config.OUTPUT_DIR / "schedule_block_sensitivity.csv")
    print(sens.round(1).to_string())

    # 7 figures (wide set for slides, column set for the IEEE report)
    for mode in ("wide", "column"):
        plots.set_mode(mode)
        make_figures(scores, k, train, train_labels, names, Xtr, labels, summary, sens, daily, cm, test, starts)


    facts = {
        "k": k, "n_train_days": len(train), "n_test_days": len(test),
        "ari_kmeans_vs_ward": round(ari_ward, 3), "ari_raw_vs_shape": round(ari_shape, 3),
        "silhouette_test": round(sil_test, 3),
        "test_cluster_share_pct": test_share.round(1).to_dict(),
        "best_start_by_year": drift.to_dict(),
        "bootstrap_saving_kg_per_MWh_vs_evening": {k_: [round(v, 2) for v in val] for k_, val in boot.items()},
    }
    (config.OUTPUT_DIR / "part3_facts.json").write_text(json.dumps(facts, indent=2, default=str))
    with open(config.OUTPUT_DIR / "part3_summary.txt", "w") as f:
        f.write("PART 3 - DAY-TYPE CLUSTERING AND SCHEDULING BENEFIT\n\n")
        f.write(json.dumps(facts, indent=2, default=str) + "\n\n")
        f.write("Cluster description (training days)\n" + desc.round(1).to_string() + "\n\n")
        f.write("Day-type classifier (test)\n" + clf_metrics.round(3).to_string() + "\n\n")
        f.write(f"Scheduling summary, {config.BLOCK_HOURS}-h block, test period\n" + summary.round(2).to_string() + "\n\n")
        f.write("Saving vs evening BAU (%) by block length\n" + sens.round(1).to_string() + "\n")
    print(json.dumps(facts, indent=2, default=str))
    print(f"\nAll outputs in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
