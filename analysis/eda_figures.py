"""Exploratory heat maps for the report: fuel mix, feature correlation, and missing data.

- fig_fuel_mix.pdf: mean share of positive generation by fuel, by calendar month (2019-2025),
  from the raw EIA-930 fuel-type file, over hours where the eight fuels other than geothermal
  are all reported. Negative values are clipped to zero, as in the CI target.
- fig_feature_corr.pdf: Pearson correlation of the 26 headline features and the lifecycle target,
  from the frozen 7-day table.
- fig_missing.pdf: missing fuel-type hours by year and month, by fuel, from the raw file on the
  complete hourly UTC grid.

Writes analysis/eda_summary.json with the numbers the report cites.

    python3 -m analysis.eda_figures
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.loader import load_feature_table

RAW = Path("data/raw/eia930/eia930_CISO_fuel_type_2019-2025.csv")
TABLE = "data/processed/h168_v0.3.0"
FIG_DIR = Path("termreport/figures")
SUMMARY = Path("analysis/eda_summary.json")
FUELS = ["COL", "NG", "NUC", "OIL", "SUN", "WAT", "WND", "OTH", "GEO"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_fuel_hourly():
    raw = pd.read_csv(RAW)
    raw["t"] = pd.to_datetime(raw["period"], utc=True)
    grid = pd.date_range("2019-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
    wide = raw.pivot_table(index="t", columns="fueltype", values="value", aggfunc="first")
    wide = wide.reindex(grid)
    return wide.reindex(columns=FUELS)


MIX_FUELS = [f for f in FUELS if f != "GEO"]  # geothermal reports only 375 hours


def fuel_mix(wide, summary):
    mix = wide[MIX_FUELS]
    complete = mix.notna().all(axis=1)  # hours where all eight fuels are reported
    pos = mix[complete].clip(lower=0)
    share = pos.div(pos.sum(axis=1), axis=0)
    by_month = share.groupby(share.index.month).mean()  # rows: 1..12
    mat = by_month.T.reindex(MIX_FUELS).to_numpy() * 100
    summary["fuel_mix_complete_hours"] = int(complete.sum())
    summary["fuel_mix_total_hours"] = int(len(wide))
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS, fontsize=7)
    ax.set_yticks(range(len(MIX_FUELS)))
    ax.set_yticklabels(MIX_FUELS, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v < 35 else "black")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Share of positive generation (%)", fontsize=7)
    ax.set_xlabel("Month (2019-2025 average)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_fuel_mix.pdf")
    plt.close(fig)
    summary["fuel_share_pct_annual"] = {f: round(float(share[f].mean() * 100), 2) for f in MIX_FUELS}
    summary["fuel_share_pct_by_month"] = {
        f: [round(float(v), 2) for v in mat[i]] for i, f in enumerate(MIX_FUELS)
    }


def feature_corr(summary):
    df, features, _ = load_feature_table(TABLE)
    cols = list(features) + ["target_ci_lifecycle"]
    corr = df[cols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [c.replace("target_ci_lifecycle", "TARGET") for c in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels, fontsize=5.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label("Pearson correlation", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_feature_corr.pdf")
    plt.close(fig)
    tcol = corr[-1, :-1]
    order = np.argsort(-np.abs(tcol))
    summary["target_corr_top5"] = [[features[i], round(float(tcol[i]), 3)] for i in order[:5]]
    summary["max_abs_corr_between_features"] = round(float(np.max(np.abs(np.triu(corr[:-1, :-1], 1)))), 3)


def missing_map(wide, summary):
    miss_all = wide.isna().astype(int)
    total = miss_all.sum()
    summary["missing_hours_by_fuel"] = {f: int(total[f]) for f in FUELS}
    summary["missing_share_pct_by_fuel"] = {f: round(float(total[f]) / len(wide) * 100, 2) for f in FUELS}
    # Geothermal is left off the map: it is missing in 99% of hours and saturates the colour scale.
    miss = miss_all[MIX_FUELS]
    by_ym = miss.groupby([miss.index.year, miss.index.month]).sum()
    years = sorted(set(by_ym.index.get_level_values(0)))
    mat = np.full((len(years) * 12, len(MIX_FUELS)), np.nan)
    for r, (y, m) in enumerate([(y, m) for y in years for m in range(1, 13)]):
        if (y, m) in by_ym.index:
            mat[r] = by_ym.loc[(y, m)].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.imshow(np.log10(mat + 1), aspect="auto", cmap="magma_r")
    ax.set_xticks(range(len(MIX_FUELS)))
    ax.set_xticklabels(MIX_FUELS, fontsize=7)
    ticks = list(range(0, len(years) * 12, 3))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{years[t // 12]}-{MONTHS[t % 12]}" for t in ticks], fontsize=6)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Missing hours (log10 of count + 1)", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_missing.pdf")
    plt.close(fig)
    worst = by_ym.sum(axis=1).sort_values(ascending=False).head(3)
    summary["worst_months_missing_hours"] = {f"{y}-{m:02d}": int(v) for (y, m), v in worst.items()}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    wide = load_fuel_hourly()
    summary = {}
    fuel_mix(wide, summary)
    feature_corr(summary)
    missing_map(wide, summary)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1)[:1500])


if __name__ == "__main__":
    main()
