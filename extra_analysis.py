"""
extra_analysis.py - per-month capture and per-day saving distribution (report Sec. VI).

    python extra_analysis.py      (run after run_part3.py)
"""

import numpy as np
import pandas as pd

import config
import plots
from plots import INK2, SERIES, _col, _fs, _save, plt


def main():
    d = pd.read_csv(config.OUTPUT_DIR / "schedule_daily_emissions.csv", index_col=0, parse_dates=True)
    m = d.groupby(d.index.to_period("M")).mean()
    cap = pd.DataFrame({c: (m.evening_bau - m[c]) / (m.evening_bau - m.oracle)
                        for c in ["static_tou", "cluster_pred", "cluster_adaptive"]})
    cap.index = cap.index.to_timestamp()
    cap.round(3).to_csv(config.OUTPUT_DIR / "monthly_capture.csv")

    sav = pd.DataFrame({"adaptive": d.evening_bau - d.cluster_adaptive,
                        "fixed": d.evening_bau - d.cluster_pred,
                        "regret_adaptive": d.cluster_adaptive - d.oracle,
                        "regret_fixed": d.cluster_pred - d.oracle})
    stats = {
        "share_days_adaptive_worse_than_bau": float((sav.adaptive < 0).mean()),
        "share_days_adaptive_equals_oracle": float((sav.regret_adaptive < 1e-9).mean()),
        "share_days_fixed_equals_oracle": float((sav.regret_fixed < 1e-9).mean()),
        "median_saving_adaptive": float(sav.adaptive.median()),
        "p10_saving_adaptive": float(sav.adaptive.quantile(0.1)),
        "p90_saving_adaptive": float(sav.adaptive.quantile(0.9)),
        "median_regret_adaptive": float(sav.regret_adaptive.median()),
        "p90_regret_adaptive": float(sav.regret_adaptive.quantile(0.9)),
        "median_regret_fixed": float(sav.regret_fixed.median()),
        "p90_regret_fixed": float(sav.regret_fixed.quantile(0.9)),
    }
    pd.Series(stats).round(3).to_csv(config.OUTPUT_DIR / "daily_saving_stats.csv")
    print(pd.Series(stats).round(3))
    print(cap.round(2).to_string())

    for mode in ("wide", "column"):
        plots.set_mode(mode)
        fig, ax = plt.subplots(figsize=_fs(7, 2.6))
        for c, col, lab in [("static_tou", SERIES[3], "Static TOU"), ("cluster_pred", SERIES[2], "Day type, fixed"),
                            ("cluster_adaptive", SERIES[1], "Day type, adaptive")]:
            ax.plot(cap.index, cap[c], marker="o", lw=1.5, ms=3 if _col() else 5, color=col, label=lab)
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.set(ylabel="Capture of oracle saving", title="Monthly capture of the oracle saving, 2024-2026",
               ylim=(-0.3, 1.08))
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(ncol=3, loc="lower center", fontsize=None if _col() else 7)
        _save(fig, "fig_monthly_capture.png")

        fig, ax = plt.subplots(figsize=_fs(7, 2.6))
        bins = np.arange(0, 41, 2)
        ax.hist(sav.regret_fixed.clip(upper=40), bins=bins, color=SERIES[2], alpha=0.75, label="Day type, fixed")
        ax.hist(sav.regret_adaptive.clip(upper=40), bins=bins, color=SERIES[1], alpha=0.75, label="Day type, adaptive")
        ax.set(xlabel="Daily regret vs oracle (kg CO$_2$/MWh, clipped at 40)", ylabel="Test days",
               title="Distribution of daily regret, W = 4 h")
        ax.legend(fontsize=None if _col() else 7)
        _save(fig, "fig_regret_hist.png")


if __name__ == "__main__":
    main()


def drift_within_types():
    """Training centroid vs test-period mean profile for each day type, and weekday/weekend savings."""
    lab = pd.read_csv(config.OUTPUT_DIR / "day_labels.csv", index_col=0, parse_dates=True)["cluster"]
    import data_prep
    days = data_prep.daily_matrix(data_prep.load_hourly())
    lab = lab.reindex(days.index)
    tr = days.index <= pd.Timestamp(config.TRAIN_END)
    rows = []
    for mode in ("wide", "column"):
        plots.set_mode(mode)
        fig, axes = plt.subplots(1, 3, figsize=_fs(7, 2.4), sharey=True)
        for c, ax in enumerate(axes):
            a = days[tr & (lab == c).values].mean()
            b = days[(~tr) & (lab == c).values].mean()
            ax.plot(range(24), a.values, color=SERIES[c], lw=1.5, ls="--", label="Train centroid")
            ax.plot(range(24), b.values, color=SERIES[c], lw=2, label="Test mean")
            ax.set(title=f"C{c}", xticks=range(0, 24, 6), xlabel="Hour")
            if c == 0:
                ax.set_ylabel("gCO$_2$/kWh")
            if mode == "wide":
                rows.append({"cluster": c, "train_cleanest_hour": int(a.values.argmin()),
                             "test_cleanest_hour": int(b.values.argmin()),
                             "train_best4_start": int(np.argmin(np.convolve(a.values, np.ones(4), "valid"))),
                             "test_best4_start": int(np.argmin(np.convolve(b.values, np.ones(4), "valid")))})
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=None if _col() else 8)
        fig.suptitle("Day types: training centroid vs. 2024-2026 mean profile", fontsize=None if _col() else 10)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(plots.MODE["dir"] / "fig_type_drift.png", dpi=300 if _col() else 200)
        plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(config.OUTPUT_DIR / "type_drift.csv", index=False)
    print(out)

    d = pd.read_csv(config.OUTPUT_DIR / "schedule_daily_emissions.csv", index_col=0, parse_dates=True)
    wk = np.where(d.index.dayofweek >= 5, "weekend", "weekday")
    g = d.groupby(wk).mean()
    res = pd.DataFrame({"n": d.groupby(wk).size(), "bau": g.evening_bau,
                        "adaptive_saving_kg": g.evening_bau - g.cluster_adaptive,
                        "adaptive_saving_pct": 100 * (g.evening_bau - g.cluster_adaptive) / g.evening_bau,
                        "oracle_saving_pct": 100 * (g.evening_bau - g.oracle) / g.evening_bau}).round(2)
    res.to_csv(config.OUTPUT_DIR / "weekday_weekend.csv")
    print(res)


if __name__ == "__main__":
    drift_within_types()
