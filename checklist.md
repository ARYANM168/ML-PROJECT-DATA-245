# Data Pipeline Checklist

Jayanth Naga Sai Pasupulati (M1, data pipeline)
Today: Nov 18 (Day 28). Final submission: Dec 1 (Day 37).

## Before you start

- [ ] Confirm with Aryan and Jaskirat which tasks are done, and tell them your realistic date
- [ ] Agree on balancing authorities (CA ISO only, or 6–8 regions)
- [ ] Agree on forecast horizon (7-day-ahead, or next hour, 6 hours, and a day)
- [ ] Create your branch in the private GitHub repo; add .gitignore for raw data, processed data, and API keys
- [ ] Set the NOAA key as an environment variable, not in code
- [ ] Install PUDL (conda-forge: catalystcoop.pudl), Python, pandas, pyarrow; pin versions in requirements
- [ ] Download EIA-930 data for 2019–2025; record PUDL version, download date, and SHA-256 checksums; make raw files read-only
- [ ] Pull NOAA temperature, wind, and solar data for the same years; confirm solar is available
- [ ] Inventory raw data: missing hours by BA and fuel type, negative values, values above nameplate

## Urgent: pipeline is on the critical path

- [ ] Freeze the feature table (25–30 features) and hand it to M2 and M3
- [ ] Confirm M2 and M3 can load the feature table

## Data cleaning

- [x] Drop days with more than 10% missing hours
- [ ] Impute remaining gaps with PUDL-recommended methods; flag every imputed value
- [x] Flag outliers (generation above nameplate, negative values) without deleting them
- [x] Store timestamps in UTC; add a local-time column and a DST flag

## Target and features

- [x] Compute CI(t) = Σ(generation × emission factor) / total generation
- [ ] Use IPCC AR5 lifecycle medians; add combustion-only sensitivity run
- [x] Unit-test the CI formula against hand-computed examples
- [x] Add lags t-1, t-2, t-24, t-48, t-168
- [x] Add hour sine/cosine, day-of-week, month, day-of-year, is_weekend
- [x] Add 6-hour and 24-hour rolling mean and std of CI and demand

## Weather

- [ ] Area-weight NOAA data to each balancing authority
- [x] Gap-fill with climatological normals; flag filled values

## Splits and validation

- [x] Train 2019–2023, validate early 2024, test late 2024–2025
- [x] Rolling-origin validation: weekly, forecasting 7 days ahead
- [ ] Confirm no future data enters training
- [ ] Record split dates and observation counts

## Deliverables

- [ ] Reproducible pipeline code, merged via pull requests
- [ ] Data quality report (missing values, imputation rates, outliers, DST, coverage)
- [ ] EDA notebook (seasonal patterns, fuel mix, correlations)
- [ ] Data dictionary, provenance record, README datasheet, datapackage.json
- [ ] Final report section and presentation (shared tasks Q and R)

## Release (after grading)

- [ ] Check EIA, NOAA, and PUDL license terms
- [ ] Get written team agreement to CC-BY 4.0
- [ ] Publish to Zenodo: CSV and Parquet, checksum manifest, link to GitHub release
- [ ] Fix the DMP release date (currently 2026-10-31, before the Dec 1 report)

## Resolve before submission

- [ ] Align scope: PERT says CA ISO only; proposal and DMP say 6–8 BAs
- [ ] Align horizon: proposal says 7-day-ahead; PERT says next hour, 6 hours, and a day
- [ ] Use UTC storage (per DMP), not local time (per proposal)
- [ ] Add IPCC AR5 emission factors to the proposal or report
- [ ] Align cost statement: proposal says "only my time"; DMP allows up to about $12

## Progress notes (2026-11-18)

Code is in `pipeline/` with 32 passing tests in `tests/` (run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest`). Ticked items are implemented and tested, not yet run on real data.

Not done, and why:
- Team items (confirm tasks, agree BAs and horizon, set branch in GitHub): need the teammates or the GitHub repo.
- NOAA key and EIA-930 download: no NOAA or EIA key is set in this environment, and no raw data is downloaded yet.
- PUDL install: not installed. Use conda-forge (catalystcoop.pudl) and record the version.
- Checksums, read-only raw files, inventory, and station-to-BA weights: helpers exist (`pipeline/provenance.py`, `pipeline/inventory.py`, `pipeline/weather.py`) but have not run on data.
- PUDL-recommended imputation: the code uses linear interpolation up to 3 hours. Replace it with the PUDL method if one applies.
- IPCC AR5 and combustion-only factors: `pipeline/emissions.py` holds working values from memory. Verify them against the published tables before citing them.
- Split choice: validate is Jan-Jun 2024 and test is Jul 2024-Dec 2025. The checklist does not say where Apr-Jun 2024 goes. Confirm with the team.
