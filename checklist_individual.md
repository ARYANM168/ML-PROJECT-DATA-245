# Individual Checklist: Coding (M1, Data Pipeline)

Jayanth Naga Sai Pasupulati
Updated: Nov 18 (Day 28). Final submission: Dec 1 (Day 37).

Scope: code, data processing, and tests only. Documentation and report items are tracked elsewhere.

Status key: [x] done, [~] partly done, [ ] open, [!] blocked on a decision or input from someone else.

## Setup

- [x] NOAA CDO token stored as environment variable (in `.env`, gitignored)
- [x] EIA API key stored as environment variable (in `.env`, gitignored)
- [x] PUDL installed in its own conda environment (`catalystcoop.pudl` 2025.12.1)
- [~] Git repo: not initialized in this folder yet (`git init`, first commit, branch)
- [~] `.gitignore` written (raw data, processed data, parquet, `.env`, secrets, caches)
- [x] `requirements.txt` pinned for the project environment (pandas 3.1.0, numpy 2.2.5, pyarrow 24.0.0, pytest 9.0.2, scikit-learn 1.6.1)

## Data acquisition

- [x] `pipeline/download.py`: EIA-930 region and fuel-type downloads (CAISO, 2019-2025), paginated
- [x] `pipeline/download.py`: NOAA GHCND daily downloads (LAX, SAN, SFO, SMF, FAT, 2019-2025)
- [x] `record_download` on every raw file: SHA-256, source URL without key, date, read-only (mode 0444)
- [x] Manifest verification passes for all 7 raw files (`pipeline/provenance.py`)
- [x] Raw inventory run and saved (`reports/inventory_eia930_CISO_2026-11-18.csv`)
- [x] NOAA coverage summary saved (`reports/noaa_coverage_2026-11-18.csv`)
- [x] Download EIA generator capacity for nameplate values (December EIA-860 snapshots, CA, 2019-2025; `pipeline/capacity.py`)
- [x] Join nameplate to the fuel-type data and run the above-nameplate check on real data (0 hours above nameplate for any fuel with a capacity mapping; OTH excluded because it is not one technology)
- [x] Hourly weather from NOAA Global Hourly (ISD), 2019-2025, for the five stations (`pipeline/noaa_hourly.py`); downloads checksummed and read-only

## Cleaning and flags (`pipeline/cleaning.py`, `pipeline/run_pipeline.py`)

- [x] Full hourly UTC grid, so missing hours are explicit
- [x] Gaps up to 3 hours linearly interpolated, with a flag per fuel and for demand
- [x] 10% sparse-day rule applied, with dropped days recorded in the manifest
- [x] Wire outlier flags into the pipeline output: negative generation per fuel and above-nameplate (`flag_outliers`); values are never deleted
- [x] Wire DST and local-time columns into the pipeline output (`local_time`, `dst_flag`)
- [x] Add a `ci_mapped_share` column (share of positive generation covered by mapped fuels) to the output

## Target and features (`pipeline/carbon_intensity.py`, `pipeline/features.py`)

- [x] CI(t) = sum(generation x factor) / total generation, with negatives clipped and unmapped fuels excluded
- [x] Unit tests for CI on hand-computed examples
- [x] Lifecycle target and combustion-only sensitivity target, both built
- [x] Emission factors checked against IPCC AR5 medians via Wikipedia's reproduction of Table A.III.2; oil removed (not in the table)
- [~] Emission factors: primary IPCC PDF is blocked (403). Add the PDF to the project folder to close this. Combustion-only factors are assumptions (heat rates) and remain flagged in the code
- [x] 7-day feature builder (`build_features_lead`): every input at least 168 hours old, 26 features
- [x] 1-hour feature builder (`build_features`) with previous-day weather, 27 features
- [x] Hydro rule: CI kept where only hydro is missing, flagged in `ci_hydro_missing`

## Weather (`pipeline/weather.py`)

- [x] Station-to-BA weights in `config/station_weights.csv` (CA ISO only, equal weights, team decision)
- [x] Area-weighted averaging that renormalizes over stations reporting each day
- [x] Day-of-year climatology gap-fill from before 2024-01-01, flagged
- [x] Weather features use hour t-1 and the 24 hours before t (no same-hour leakage)
- [x] Weather excluded from the 7-day model (no forecast weather available)
- [x] Solar driver: dropped (no NOAA solar data). Team confirmed (2026-11-18)

## Splits and validation (`pipeline/splits.py`, `pipeline/model.py`)

- [x] Train 2019-2023, validate Jan-Jun 2024, test Jul 2024-Dec 2025 (UTC)
- [x] Split dates and observation counts recorded in the manifest
- [x] Weekly rolling-origin evaluation for the 7-day model (74 folds on the test period)
- [x] Persistence and seasonal 4-week baselines implemented
- [x] No-future-data check: passes for both horizons (40/40 samples), with a positive control (40/40)
- [x] Horizon: 7-day headline. Team confirmed (2026-11-18)

## Pipeline and outputs (`pipeline/run_pipeline.py`)

- [x] One command rebuilds the feature table from raw files (`--horizon 168` or `--horizon 1`)
- [x] Parquet output with SHA-256 manifest and version tag
- [x] Frozen 7-day table: `data/processed/h168_v0.2.0/` (61,272 rows, 26 features)
- [x] Secondary 1-hour table: `data/processed/h1_v0.2.0/` (61,272 rows, 27 features)
- [x] Feature-count guard (25-30) before freezing
- [x] Smoke run on a small date range (October 2024)
- [x] Reproducibility check: rebuilding the 7-day table from raw files gives the same SHA-256 (`91bda3be...`)

## Tests (`tests/`)

- [x] 38 tests passing: CI, cleaning, features (both horizons), splits, provenance, inventory, weather, no-future check
- [x] End-to-end smoke test in the suite (October 2024, horizon 1; asserts feature count and the new columns)
- [x] Tests for `flag_outliers` and `add_time_columns` (unit tests exist; the smoke test checks they are wired in)

## Handoff to M2 and M3 (code side)

- [x] Loader check script (`python3 -m pipeline.check_table <table_dir>`): verifies checksum, index, features, targets; both tables PASS
- [x] Loader helper for M2 and M3 (`pipeline/loader.py`: `load_feature_table(dir)` verifies the checksum and returns features, targets, and split labels)
- [x] Loads checked in both environments: project (pandas 3.1.0) and `pudl` (pandas 2.3.3, pyarrow 20.0.0); checksums match
- [!] Teammates confirm they can load the table in their own code (needs M2 and M3)

## Version control and environment

- [ ] `git init`, first commit of `pipeline/`, `tests/`, `config/`, `requirements.txt`, `.gitignore`, `checklist_individual.md` (exclude `.env`, `data/`, `reports/` if large)
- [ ] Create the branch in the private GitHub repo, push, and open a PR
- [ ] Note: the PUDL environment uses pandas 2.3.3, the project environment uses pandas 3.1.0. Keep PUDL steps in the `pudl` environment if they are added

## Decisions still needed (not coding)

- [x] Confirm the 7-day horizon as the headline with the team (confirmed 2026-11-18)
- [x] Confirm the hydro rule (CI kept where only hydro is missing) (confirmed 2026-11-18)
- [x] Confirm dropping the solar driver (confirmed 2026-11-18)

## Outlier and coverage summary (7-day table, 2019-2025)

- Negative hours flagged: solar 24,980; coal 5,017; oil 1,581; nuclear 446; wind 24; hydro 185; OTH 27,563 (OTH has mixed sign and is not one technology; flagged, not interpreted).
- Above nameplate: 0 hours for any fuel with a capacity mapping.
- CI mapped share: 1.0 wherever CI is computed.
- DST hours (Pacific daylight time): 39,944 of 61,272.

## Results to date (7-day headline, test period, MAE in gCO2/kWh)

| Target | Model (rolling origin, 74 weekly folds) | Seasonal 4-week | Persistence 168h |
|---|---|---|---|
| Lifecycle | 34.97 | 36.66 | 41.15 |
| Combustion-only | 29.84 | 31.17 | 34.91 |

The single fit trained on 2019-2023 is worse than the seasonal average (39.94 vs 36.77 lifecycle), so the rolling-origin result is the one to use.

1-hour secondary model, fixed split: lifecycle 11.36 (1-hour persistence 11.80); combustion-only 10.36 (10.85).

Hydro-missing hours (7,907) are all in 2019-2020, which is training data. The test period has no hydro gaps, so the hydro rule affects training only.


## Hourly weather (2026-11-18)

- Source: NCEI Global Hourly (ISD) station files. Temperature and wind, averaged across the five stations with equal weights.
- Quality: 99.9% of hours have temperature and wind for every station-year, except the 2025 files, which end on 2025-08-27.
- The 2025 gap (3,041 hours, late August to December) is filled with hour-of-day climatology from before 2024-01-01 and flagged (`wx_any_imputed`). The 1-hour model's metrics are also reported on observed-weather rows only.
- Station check: the GHCN station USW00023232 ("Sacramento Airport") is Sacramento Executive (ISD 72483023232), not Sacramento International. Both sources use the same airport, so the mapping is consistent, but the report should name it correctly.
- Result: hourly weather does not improve the 1-hour model. Test MAE lifecycle 11.42 (daily weather version: 11.36); observed-weather rows 11.56. Report hourly weather as a tested but non-helpful input.
- Tables: headline `h168_v0.3.0` (26 features, no weather); secondary `h1_v0.3.0` (28 features, hourly weather). The `v0.2.0` tables are superseded.
