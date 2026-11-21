# Provenance Notes

## Software environment

| Item | Version | Source |
|---|---|---|
| catalystcoop.pudl | 2025.12.1 | conda-forge (`conda create -n pudl -c conda-forge catalystcoop.pudl`) |
| pandas (pudl env) | 2.3.3 | conda-forge |
| pyarrow (pudl env) | 20.0.0 | conda-forge |
| Python (pudl env) | 3.13 | conda-forge |

Environment location: `/opt/anaconda3/envs/pudl`
Recorded: 2026-11-18

## Project environment (pipeline code and tests)

Pinned in `requirements.txt`: pandas 3.1.0, numpy 2.2.5, pyarrow 24.0.0, pytest 9.0.2.

Note: the PUDL environment uses pandas 2.3.3 and the project environment uses pandas 3.1.0. Run PUDL-dependent steps in the `pudl` environment, and run the feature code in the project environment, or pin one pandas version for both before the final run.

## Data sources

| Source | Access | Status |
|---|---|---|
| EIA-930 hourly, CAISO (region + fuel type) | EIA API key (in `.env`, `EIA_API_KEY`) | downloaded 2026-11-18; checksums in `data/raw/manifest.json` |
| NOAA CDO GHCND daily (LAX, SAN, SFO, SMF, FAT) | CDO token (in `.env`, `NOAA_CDO_TOKEN`) | downloaded 2026-11-18 (TMAX, TMIN, AWND); TAVG and TSUN not returned |
| Solar irradiance | none yet | not in GHCND; needs another source (e.g., NREL NSRDB) |
| Hourly weather | none yet | NOAA CDO is daily only; needs ISD/LCD hourly archive |
| EIA generator capacity (nameplate) | none yet | needed for the above-nameplate check |

Raw data files: `data/raw/` (gitignored). Dated reports: `reports/` (not gitignored).
