"""Split the 74 fold results by origin year and summarize each part.

Reads analysis/fold_stats.json (written by analysis/fold_stats.py); no model is refit.
Writes analysis/subperiod_stats.json.

    python3 -m analysis.subperiod_stats
"""

import json
from pathlib import Path

import numpy as np

IN = Path("analysis/fold_stats.json")
OUT = Path("analysis/subperiod_stats.json")
N_BOOT = 5000
SEED = 0


def summarize(rows, rng):
    diff = np.array([r["model_mae"] - r["seasonal_4w_mae"] for r in rows])
    means = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(N_BOOT)])
    return {
        "folds": int(len(rows)),
        "model_beats_seasonal": int((diff < 0).sum()),
        "mean_fold_diff": float(diff.mean()),
        "bootstrap_95ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def main():
    data = json.load(open(IN))["targets"]
    out = {}
    for tgt, block in data.items():
        rows = block["per_fold"]
        out[tgt] = {}
        for label, keep in [("origins_2024", lambda r: r["origin"].startswith("2024")),
                            ("origins_2025", lambda r: r["origin"].startswith("2025")),
                            ("all", lambda r: True)]:
            rng = np.random.default_rng(SEED)
            out[tgt][label] = summarize([r for r in rows if keep(r)], rng)
        print(tgt, json.dumps(out[tgt], indent=1))
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
