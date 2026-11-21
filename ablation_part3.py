"""
ablation_part3.py - robustness checks for Part 3.

    python ablation_part3.py

For each representation (raw gCO2/kWh profiles vs. shape-normalised profiles)
and each k in {3, 4, 6, 8}, the script reports:
  * silhouette on training days
  * the share of the oracle saving captured when the TRUE day type of the test
    day is known and its fixed training centroid sets the 4-hour block
    (an upper bound for fixed-centroid day-type scheduling)
Results: outputs/ablation_representation_k.csv
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import clustering
import config
import data_prep
import scheduling as S


def main():
    days = data_prep.daily_matrix(data_prep.load_hourly())
    train, test = data_prep.split_train_test(days)
    W = config.BLOCK_HOURS
    act = test.to_numpy()
    bau = np.mean([S.emissions_for_start(a, min(config.EVENING_START, 24 - W), W) for a in act])
    orc = np.mean([S.block_sums(a, W).min() / W for a in act])
    rows = []
    for rep in ["raw", "shape"]:
        Xtr = train.to_numpy() if rep == "raw" else clustering.shape_normalise(train.to_numpy())
        Xte = act if rep == "raw" else clustering.shape_normalise(act)
        for k in [3, 4, 6, 8]:
            km = KMeans(k, n_init=config.N_INIT, random_state=config.RANDOM_STATE).fit(Xtr)
            lab = km.predict(Xte)
            best = [S.best_start(c, W) for c in km.cluster_centers_]
            e = np.mean([S.emissions_for_start(act[i], best[lab[i]], W) for i in range(len(act))])
            rows.append({"representation": rep, "k": k,
                         "train_silhouette": silhouette_score(Xtr, km.labels_),
                         "distinct_block_starts": len(set(best)),
                         "saving_vs_evening_pct": 100 * (bau - e) / bau,
                         "capture_of_oracle": (bau - e) / (bau - orc)})
    out = pd.DataFrame(rows).round(3)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out.to_csv(config.OUTPUT_DIR / "ablation_representation_k.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
