"""End-to-end build: raw files -> feature table (Parquet) with checksum, version, and reports.

Examples:
    python -m pipeline.run_pipeline --start 2024-10-01 --end 2024-10-31 --tag test_small --horizon 1 --skip-model
    python -m pipeline.run_pipeline --start 2019-01-01 --end 2025-12-31 --tag h168_v0.2.0 --horizon 168
    python -m pipeline.run_pipeline --start 2019-01-01 --end 2025-12-31 --tag h1_v0.2.0 --horizon 1

Headline model: --horizon 168 (7 days ahead). Every feature is at least 168 hours old, and
there is no weather (forecast weather is not available). Secondary check: --horizon 1, which
uses hourly NOAA weather (ISD) observed before each hour.

Steps:
 1. Hourly UTC grid. Fuel mix pivoted wide; demand from the region file (type D).
 2. Short gaps (up to 3 hours) linearly interpolated per fuel and for demand, each flagged.
 3. CI from the mapped fuels (GEO excluded: 375 non-missing rows in the EIA feed).
    Hours where only hydro (WAT) is missing keep their CI and are flagged in ci_hydro_missing.
    Any other mapped fuel still missing after imputation makes CI NaN.
 4. Weather (horizon 1 only): hourly temperature and wind from ISD, equal-weighted across the
    stations in config/station_weights.csv. Gaps up to 3 hours interpolated, longer gaps filled
    with hour-of-day climatology from before 2024-01-01. All flagged. Features use hour t-1 and
    the 24 hours before t.
 5. Features computed on the full hourly grid, then rows of days with more than 10% missing
    CI hours are dropped.
 6. Write Parquet and a SHA-256 manifest; run the no-future check; fit baselines and models on
    both targets; for horizon 168, also run weekly rolling-origin evaluation on the test period.
"""

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .carbon_intensity import carbon_intensity
from .checks import no_future_check
from .capacity import capacity_by_year
from .cleaning import add_time_columns, drop_sparse_days, flag_outliers, impute_short_gaps
from .emissions import COMBUSTION_ONLY_G_PER_KWH, LIFECYCLE_AR5_G_PER_KWH
from .features import build_features, build_features_lead, build_hourly_weather_features
from .model import add_mase, fit_and_score, naive_scale, rolling_origin_eval
from .splits import split_masks, split_summary
from .noaa_hourly import load_hourly_weather
from .weather import fill_with_climatology

FUEL_FILE = Path("data/raw/eia930/eia930_CISO_fuel_type_2019-2025.csv")
REGION_FILE = Path("data/raw/eia930/eia930_CISO_region_demand_gen_2019-2025.csv")
NOAA_GLOB = "data/raw/noaa/ghcnd_*.csv"
WEIGHTS_FILE = Path("config/station_weights.csv")
CLIM_REFERENCE_END = "2024-01-01"
REPORTS = Path("reports")
MAX_GAP_HOURS = 3
SPARSE_DAY_MAX_MISSING = 0.10
GEO = "GEO"
HYDRO = "WAT"
HYDRO_FLAG = "ci_hydro_missing"
LEAD_WEEK = 168
ROLLING_START, ROLLING_END = "2024-07-01", "2025-12-31 23:00"


def hourly_grid(start, end):
    return pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h", tz="UTC")


def load_fuel_wide(grid):
    df = pd.read_csv(FUEL_FILE)
    ts = pd.to_datetime(df["period"], utc=True)
    wide = df.assign(ts=ts).pivot_table(index="ts", columns="fueltype", values="value", aggfunc="first")
    return wide.reindex(grid)


def load_demand(grid):
    df = pd.read_csv(REGION_FILE)
    df = df[df["type"] == "D"]
    ts = pd.to_datetime(df["period"], utc=True)
    s = pd.Series(df["value"].to_numpy(), index=ts).groupby(level=0).first()
    return s.reindex(grid).rename("demand_mw")


def build_inputs(start, end):
    grid = hourly_grid(start, end)
    record = {"grid_hours": int(len(grid))}

    fuel = load_fuel_wide(grid)
    raw_fuel = fuel.copy()  # before imputation, so outlier flags see the reported values
    mapped = [c for c in fuel.columns if c in LIFECYCLE_AR5_G_PER_KWH and c != GEO]
    fuel_flags = {}
    for c in mapped:
        fuel[c], fuel_flags[c] = impute_short_gaps(fuel[c], limit=MAX_GAP_HOURS)

    required = [c for c in mapped if c != HYDRO]  # hydro gaps are allowed and flagged
    complete = fuel[required].notna().all(axis=1)

    life_factors = {c: LIFECYCLE_AR5_G_PER_KWH[c] for c in mapped}
    comb_factors = {c: COMBUSTION_ONLY_G_PER_KWH[c] for c in mapped}
    life = carbon_intensity(fuel[mapped], life_factors)
    comb = carbon_intensity(fuel[mapped], comb_factors)
    ci_life = life["ci_g_per_kwh"].where(complete).rename("target_ci_lifecycle")
    ci_comb = comb["ci_g_per_kwh"].where(complete).rename("target_ci_combustion")

    flags = {f"imputed_{c.lower()}": f for c, f in fuel_flags.items()}
    flags["ci_has_imputed_input"] = pd.concat(fuel_flags.values(), axis=1).any(axis=1)
    flags[HYDRO_FLAG] = fuel[HYDRO].isna() if HYDRO in fuel.columns else pd.Series(False, index=grid)
    flags["ci_mapped_share"] = life["mapped_share"].where(complete)

    # outlier flags: values are flagged, never deleted
    capacity = capacity_by_year()
    years = pd.Series(grid.year, index=grid)
    for c in raw_fuel.columns:
        if c in capacity.columns and c != "OTH":  # OTH is not one technology: no nameplate check
            nameplate = years.map(capacity[c]).astype(float)
            out = flag_outliers(raw_fuel[c], nameplate)
            flags[f"outlier_above_nameplate_{c.lower()}"] = out["above_nameplate"] & raw_fuel[c].notna()
        else:
            out = flag_outliers(raw_fuel[c], float("nan"))
        flags[f"outlier_negative_{c.lower()}"] = out["negative"] & raw_fuel[c].notna()

    demand_raw = load_demand(grid)
    demand, flags["imputed_demand"] = impute_short_gaps(demand_raw, limit=MAX_GAP_HOURS)

    temp, wind = load_hourly_weather(grid)
    wx = pd.DataFrame({"temp": temp, "wind": wind})
    for v in ["temp", "wind"]:
        wx[v], f_short = impute_short_gaps(wx[v], limit=MAX_GAP_HOURS)
        wx[v], f_long = fill_with_climatology(wx[v], CLIM_REFERENCE_END)
        flags[f"wx_{v}_imputed"] = f_short | f_long
    flags["wx_any_imputed"] = flags["wx_temp_imputed"] | flags["wx_wind_imputed"]

    record.update(
        {
            "mapped_fuels": mapped,
            "fuels_required_for_ci": required,
            "excluded_fuels": sorted(set(fuel.columns) - set(mapped)),
            "ci_hours_nan_after_rules": int(ci_life.isna().sum()),
            "ci_hours_hydro_missing": int(flags[HYDRO_FLAG].sum()),
        }
    )
    return grid, ci_life, ci_comb, demand, wx, flags, record


def build_panel(start, end, horizon):
    grid, ci_life, ci_comb, demand, wx, flags, record = build_inputs(start, end)

    if horizon == 1:
        feats = build_features(ci_life, demand)
        wx_feats = build_hourly_weather_features(wx["temp"], wx["wind"])
        features = pd.concat([feats, wx_feats], axis=1)
    else:
        features = build_features_lead(ci_life, demand, lead=LEAD_WEEK)

    _, dropped_days = drop_sparse_days(ci_life, max_missing_frac=SPARSE_DAY_MAX_MISSING)
    record["dropped_days"] = [str(d) for d in dropped_days]
    drop_mask = pd.DatetimeIndex(grid.tz_localize(None).normalize()).isin(pd.to_datetime(dropped_days))

    panel = features.copy()
    time_cols = add_time_columns(grid)
    panel["local_time"] = time_cols["local_time"]
    panel["dst_flag"] = time_cols["dst_flag"]
    panel["target_ci_lifecycle"] = ci_life
    panel["target_ci_combustion"] = ci_comb
    panel["demand_mw"] = demand
    for kind, s in (("lifecycle", ci_life), ("combustion", ci_comb)):
        panel[f"baseline_persistence_168h_{kind}"] = s.shift(LEAD_WEEK)
        panel[f"baseline_seasonal4w_{kind}"] = pd.concat(
            [s.shift(k * LEAD_WEEK) for k in (1, 2, 3, 4)], axis=1
        ).mean(axis=1, skipna=False)
        if horizon == 1:
            panel[f"baseline_persistence_1h_{kind}"] = s.shift(1)
    for name, s in flags.items():
        panel[name] = s.to_numpy()

    panel["split"] = "none"
    for name, m in split_masks(grid).items():
        panel.loc[m, "split"] = name

    panel = panel.loc[~drop_mask]
    record["rows_after_sparse_day_rule"] = int(len(panel))
    record["rows_dropped_sparse_day"] = int(drop_mask.sum())

    feature_cols = list(features.columns)
    return panel, feature_cols, record, wx, ci_life, demand


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def baselines_for(kind, horizon):
    base = {
        "persistence_168h": f"baseline_persistence_168h_{kind}",
        "seasonal_4w": f"baseline_seasonal4w_{kind}",
    }
    if horizon == 1:
        base = {"persistence_1h": f"baseline_persistence_1h_{kind}", **base}
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--horizon", type=int, choices=[1, 168], required=True)
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    out_dir = Path("data/processed") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    panel, feature_cols, record, wx, ci_life, demand = build_panel(args.start, args.end, args.horizon)
    n_feat = len(feature_cols)
    if not 25 <= n_feat <= 30:
        raise SystemExit(f"feature count {n_feat} is outside 25-30; not freezing")

    parquet = out_dir / "feature_table.parquet"
    panel.to_parquet(parquet, index=True)
    checksum = sha256_file(parquet)

    check = no_future_check(
        ci_life, demand, wx, lead=args.horizon, use_weather=(args.horizon == 1), n_samples=40
    )
    split_info = split_summary(panel.index).to_dict(orient="records")
    split_rows = {k: int(v) for k, v in panel.groupby("split").size().items()}

    results = {}
    if args.skip_model or not (panel["split"] == "train").any():
        results = {"skipped": "--skip-model" if args.skip_model else "no training rows in this date range"}
    else:
        for tgt, kind in [("target_ci_lifecycle", "lifecycle"), ("target_ci_combustion", "combustion")]:
            baselines = baselines_for(kind, args.horizon)
            entry = {"fixed_split": fit_and_score(panel, tgt, baselines, feature_cols)}
            if args.horizon == LEAD_WEEK:
                entry["rolling_origin_test"] = add_mase(
                    rolling_origin_eval(panel, tgt, baselines, feature_cols, ROLLING_START, ROLLING_END),
                    naive_scale(panel, tgt),
                )
            results[tgt] = entry

    manifest = {
        "version": args.tag,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date_range": [args.start, args.end],
        "horizon_hours": args.horizon,
        "parquet": str(parquet),
        "sha256": checksum,
        "rows": int(len(panel)),
        "feature_count": n_feat,
        "features": feature_cols,
        "targets": ["target_ci_lifecycle", "target_ci_combustion"],
        "split_rows_in_panel": split_rows,
        "splits": split_info,
        "no_future_check": check,
        "model_results": results,
        "build": record,
    }
    (out_dir / "feature_table.manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (REPORTS / f"feature_table_manifest_{args.tag}.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"horizon={args.horizon}h rows={len(panel)} features={n_feat} sha256={checksum[:16]}... parquet={parquet}")
    print(
        f"no-future check: {check['result']} ({check['passed_future_perturbation']}/{check['samples']} passed, "
        f"control {check['positive_control_detected']}/{check['samples']})"
    )
    for tgt, r in results.items():
        if tgt == "skipped":
            print(f"model skipped: {r}")
            continue
        fs = r["fixed_split"]["test"]
        line = f"{tgt} fixed-split test MAE model={fs['model']['mae']:.2f}"
        for k in ("persistence_1h", "persistence_168h", "seasonal_4w"):
            if k in fs:
                line += f" {k}={fs[k]['mae']:.2f}"
        print(line)
        if "rolling_origin_test" in r:
            ro = r["rolling_origin_test"]
            print(
                f"  rolling-origin test ({ro['folds']} weekly folds): model MAE={ro['model']['mae']:.2f} "
                f"persistence_168h={ro['persistence_168h']['mae']:.2f} seasonal_4w={ro['seasonal_4w']['mae']:.2f}"
            )


if __name__ == "__main__":
    main()
