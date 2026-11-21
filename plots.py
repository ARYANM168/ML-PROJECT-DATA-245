"""
plots.py - figures for the Part 3 report and slides (print-ready PNG, 200 dpi).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

import config

# Fixed categorical order (validated colour-blind-safe order), never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


# "wide" = slides (7 in), "column" = one IEEE column (3.5 in)
MODE = {"scale": 1.0, "dir": config.FIG_DIR}


def set_mode(mode: str):
    if mode == "column":
        MODE.update(scale=0.5, dir=config.OUTPUT_DIR / "figures_column")
        plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "legend.fontsize": 5.5,
                             "xtick.labelsize": 6, "ytick.labelsize": 6, "lines.markersize": 3.5})
    else:
        MODE.update(scale=1.0, dir=config.FIG_DIR)
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8,
                             "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.markersize": 6})


def _col():
    return MODE["scale"] < 1


def _fs(w, h):
    s = MODE["scale"] if w > 4 else 1.0   # small figures are already column width
    return (w * s, h * (s * 1.25 if s < 1 else 1))


def _save(fig, name):
    MODE["dir"].mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(MODE["dir"] / name, dpi=300 if MODE["scale"] < 1 else 200)
    plt.close(fig)


def k_selection(scores: pd.DataFrame, k_final: int):
    fig, axes = plt.subplots(1, 2, figsize=_fs(7, 2.6))
    ax = axes[0]
    ax.plot(scores.k, scores.kmeans_silhouette, marker="o", lw=2, color=SERIES[0], label="k-means")
    ax.plot(scores.k, scores.ward_silhouette, marker="s", lw=2, color=SERIES[1], label="Ward")
    ax.axvline(k_final, color=INK2, ls="--", lw=1)
    ax.set(xlabel="Number of day types k", ylabel="Silhouette", title="(a) Silhouette (higher is better)" if not _col() else "(a) Silhouette (higher)", xticks=list(scores.k)[::2 if _col() else 1])
    ax.legend()
    ax = axes[1]
    ax.plot(scores.k, scores.kmeans_davies_bouldin, marker="o", lw=2, color=SERIES[0], label="k-means")
    ax.plot(scores.k, scores.ward_davies_bouldin, marker="s", lw=2, color=SERIES[1], label="Ward")
    ax.axvline(k_final, color=INK2, ls="--", lw=1)
    ax.set(xlabel="Number of day types k", ylabel="Davies-Bouldin", title="(b) Davies-Bouldin (lower is better)" if not _col() else "(b) Davies-Bouldin (lower)", xticks=list(scores.k)[::2 if _col() else 1])
    ax.legend()
    _save(fig, "fig_k_selection.png")


def centroids(days: pd.DataFrame, labels: pd.Series, names: dict):
    fig, ax = plt.subplots(figsize=_fs(7, 3.2))
    hours = np.arange(24)
    for c in sorted(labels.unique()):
        member = days[labels == c]
        q25, q75 = member.quantile(0.25).values, member.quantile(0.75).values
        ax.fill_between(hours, q25, q75, color=SERIES[c], alpha=0.12, lw=0)
        ax.plot(hours, member.mean().values, lw=2, color=SERIES[c],
                label=(f"{names[c]}  n={len(member)}" if not _col()
                       else f"{names[c].split(':')[0]}: {names[c].split(':')[1].split('(')[0].strip()}, n={len(member)}"))
    ax.set(xlabel="Hour of day (local time)", ylabel="Carbon intensity (gCO$_2$/kWh)",
           title="Day-type centroids with inter-quartile bands (training days)" if not _col()
           else "Day-type centroids and IQR bands (train)", xticks=range(0, 24, 4 if _col() else 2))
    if _col():
        ax.set_ylabel("gCO$_2$/kWh")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=1)
    else:
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _save(fig, "fig_centroids.png")


def dendro(X: np.ndarray, k: int):
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=_fs(7, 2.6))
    cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, color_threshold=cut,
               above_threshold_color=INK2, no_labels=True)
    ax.axhline(cut, color=SERIES[7], ls="--", lw=1)
    ax.set(ylabel="Ward linkage distance", title=f"Ward dendrogram (last 30 merges), cut at k={k}")
    ax.grid(False)
    _save(fig, "fig_dendrogram.png")


def month_heatmap(labels: pd.Series, names: dict):
    df = pd.DataFrame({"c": labels.values, "m": pd.DatetimeIndex(labels.index).month})
    tab = pd.crosstab(df.c, df.m, normalize="columns") * 100
    fig, ax = plt.subplots(figsize=_fs(7, 2.4))
    im = ax.imshow(tab.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(len(tab)), [names[c].split(":")[0] for c in tab.index])
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 55 else INK)
    ax.grid(False)
    ax.set_title("Share of days in each day type by calendar month (%)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    _save(fig, "fig_month_heatmap.png")


def year_mix(labels: pd.Series, names: dict):
    df = pd.DataFrame({"c": labels.values, "y": pd.DatetimeIndex(labels.index).year})
    tab = pd.crosstab(df.y, df.c, normalize="index") * 100
    fig, ax = plt.subplots(figsize=_fs(7, 2.6))
    bottom = np.zeros(len(tab))
    for c in tab.columns:
        ax.bar(tab.index, tab[c], bottom=bottom, color=SERIES[c], edgecolor="white", lw=1.5,
               label=names[c].split(":")[0], width=0.7)
        bottom += tab[c].values
    ax.set(ylabel="Share of days (%)", title="Day-type mix by year (2018 and 2026 are partial years)",
           xticks=tab.index)
    ax.legend(ncol=len(tab.columns), loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7 * (1 if MODE["scale"] == 1 else 0.8))
    ax.grid(axis="x", visible=False)
    _save(fig, "fig_year_mix.png")


def strategy_bars(summary: pd.DataFrame):
    labels_map = {"evening_bau": "Evening 18-22 (BAU)", "flat": "Flat 24 h", "static_tou": "Static TOU",
                  "monthly_static": "Monthly static", "yesterday": "Yesterday's profile",
                  "cluster_pred": "Day type, fixed centroid", "cluster_known": "Known type, fixed centroid",
                  "recent_7d": "Rolling 7-day profile", "cluster_adaptive": "Day type, adaptive",
                  "oracle": "Oracle (perfect foresight)"}
    s = summary["kg_per_MWh"]
    fig, ax = plt.subplots(figsize=_fs(7, 3.3))
    colors = [INK2 if i in ("evening_bau", "flat") else (SERIES[1] if i in ("cluster_pred", "cluster_known", "cluster_adaptive") else SERIES[0])
              for i in s.index]
    ax.barh([labels_map[i] for i in s.index][::-1], s.values[::-1], color=colors[::-1], height=0.6)
    for y, (i, v) in enumerate(list(s.items())[::-1]):
        pct = summary.loc[i, "saving_vs_evening_pct"]
        ax.text(v + 2, y, f"{v:.0f}  ({pct:+.0f}%)" if i != "evening_bau" else f"{v:.0f}",
                va="center", fontsize=5.5 if _col() else 7.5, color=INK)
    ax.set(xlabel="Average intensity of the shifted load (kg CO$_2$ per MWh)" if not _col()
           else "kg CO$_2$ per MWh shifted",
           title=("Test period 2024-2026: emissions of a 1 MWh, 4-hour daily flexible block" if not _col()
                  else "1 MWh daily 4-h block, test 2024-26"),
           xlim=(0, s.max() * 1.25))
    ax.grid(axis="y", visible=False)
    _save(fig, "fig_strategy_savings.png")


def block_sensitivity(sens: pd.DataFrame):
    fig, ax = plt.subplots(figsize=_fs(7, 2.6))
    for col, c, lab in [("oracle", SERIES[0], "Oracle"), ("cluster_adaptive", SERIES[1], "Day type, adaptive"),
                        ("cluster_pred", SERIES[2], "Day type, fixed centroid"), ("static_tou", SERIES[3], "Static TOU")]:
        ax.plot(sens.index, sens[col], marker="o", lw=2, color=c, label=lab)
    ax.set(xlabel="Block length W (hours)", ylabel="Saving vs evening BAU (%)",
           title="Sensitivity of the saving to the length of the flexible block" if not _col()
           else "Saving vs. block length", xticks=sens.index)
    ax.legend(ncol=2 if _col() else 4, fontsize=None if _col() else 7, loc="lower center")
    _save(fig, "fig_block_sensitivity.png")


def yearly_saving(daily: pd.DataFrame):
    yr = daily.groupby(pd.DatetimeIndex(daily.index).year).mean()
    fig, ax = plt.subplots(figsize=_fs(3.4, 2.4))
    x = np.arange(len(yr))
    ax.bar(x - 0.27, yr["evening_bau"] - yr["cluster_pred"], 0.26, color=SERIES[2], label="Fixed centroid")
    ax.bar(x, yr["evening_bau"] - yr["cluster_adaptive"], 0.26, color=SERIES[1], label="Adaptive day type")
    ax.bar(x + 0.27, yr["evening_bau"] - yr["oracle"], 0.26, color=SERIES[0], label="Oracle")
    ax.set_xticks(x, yr.index)
    ax.set(ylabel="kg CO$_2$ saved per MWh", title="Saving vs evening BAU by test year")
    ax.legend(fontsize=6, loc="upper left")
    ax.set_ylim(0, yr[["evening_bau"]].values.max() * 0 + (yr["evening_bau"] - yr["oracle"]).max() * 1.45)
    ax.grid(axis="x", visible=False)
    _save(fig, "fig_yearly_saving.png")


def example_week(test_days: pd.DataFrame, starts: pd.DataFrame, W: int, start_date: str):
    week = test_days.loc[start_date:].iloc[:7]
    st = starts.loc[week.index]
    series = week.stack().to_numpy()
    fig, ax = plt.subplots(figsize=_fs(7, 2.6))
    t = np.arange(len(series))
    ax.plot(t, series, color=INK2, lw=1.5, label="Actual intensity")
    for i, d in enumerate(week.index):
        for col, c, off in [("evening_bau", SERIES[7], 0), ("cluster_adaptive", SERIES[0], 0)]:
            s0 = i * 24 + st.at[d, col]
            ax.axvspan(s0, s0 + W, color=c, alpha=0.35, lw=0,
                       label=None if i else ("Evening BAU block" if col == "evening_bau" else "Adaptive day-type block"))
    ax.set_xticks(np.arange(0, len(series), 24) + 12, [d.strftime("%a" if _col() else "%a %d %b") for d in week.index], fontsize=None if _col() else 7)
    ax.set(ylabel="gCO$_2$/kWh", title=f"Example test week: chosen {W}-hour blocks")
    ax.legend(fontsize=None if _col() else 7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    ax.set_title(f"Example test week (7-13 Apr 2025): chosen {W}-hour blocks", pad=22 if not _col() else 20)
    _save(fig, "fig_example_week.png")


def confusion(cm: pd.DataFrame):
    norm = cm.div(cm.sum(axis=1), axis=0).fillna(0) * 100
    fig, ax = plt.subplots(figsize=_fs(3.4, 2.8))
    ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=100)
    k = len(cm)
    ax.set_xticks(range(k), [f"C{i}" for i in range(k)])
    ax.set_yticks(range(k), [f"C{i}" for i in range(k)])
    for i in range(k):
        for j in range(k):
            v = norm.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else INK)
    ax.set(xlabel="Predicted", ylabel="True", title="Day-type classifier (row %)")
    ax.grid(False)
    _save(fig, "fig_confusion.png")
