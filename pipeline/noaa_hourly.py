"""Hourly NOAA weather from the Integrated Surface Database (Global Hourly), 2019-2025.

Source: NCEI Global Hourly files on the public S3 bucket (no token needed):
    https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{isd_id}.csv

Each station uses the same airport as the GHCN station in config/station_weights.csv:
    USW00023174 LAX -> 72295023174
    USW00023188 SAN -> 72290023188
    USW00023234 SFO -> 72494023234
    USW00023232 SMF -> 72483023232  (this is Sacramento Executive, WBAN 23232, not SMF
                                     Sacramento International; the GHCN series is the same)
    USW00093193 FAT -> 72389093193

Variables used: TMP (air temperature, C) and WND speed (m/s). ISD stores tenths and uses
9999 for missing. Only observations with a passing quality code are kept.
"""

import urllib.request
from pathlib import Path

import pandas as pd

from .download import MANIFEST, PUDL_NOTE, RAW_DIR
from .provenance import record_download
from .weather import area_weighted_mean

ISD_BASE = "https://noaa-global-hourly-pds.s3.amazonaws.com"
ISD_IDS = {
    "USW00023174": "72295023174",
    "USW00023188": "72290023188",
    "USW00023234": "72494023234",
    "USW00023232": "72483023232",
    "USW00093193": "72389093193",
}
GOOD_QUALITY = {"0", "1", "4", "5", "9"}
MISSING = 9999
WEIGHTS_FILE = Path("config/station_weights.csv")


def fetch_isd_file(isd_id, year):
    """Download one station-year file to data/raw/noaa_hourly/ once, then make it read-only."""
    out = RAW_DIR / "noaa_hourly" / f"isd_{isd_id}_{year}.csv"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ISD_BASE}/{year}/{isd_id}.csv"
    with urllib.request.urlopen(url, timeout=120) as resp:
        out.write_bytes(resp.read())
    record_download(out, url, pudl_version=PUDL_NOTE, manifest_path=MANIFEST)
    return out


def _split_value(series, field_index):
    parts = series.fillna("").str.split(",", expand=True)
    return pd.to_numeric(parts[field_index], errors="coerce"), parts


def parse_isd(path):
    """Return hourly UTC DataFrame with columns temp_c and wind_ms (NaN where missing or flagged)."""
    raw = pd.read_csv(path, low_memory=False, usecols=["DATE", "TMP", "WND"])
    ts = pd.to_datetime(raw["DATE"], utc=True)

    tmp, tparts = _split_value(raw["TMP"], 0)
    tq = tparts[1].astype(str)
    temp = (tmp / 10.0).where((tmp != MISSING) & tq.isin(GOOD_QUALITY))

    wparts = raw["WND"].fillna("").str.split(",", expand=True)
    wspd = pd.to_numeric(wparts[3], errors="coerce")
    wq = wparts[4].astype(str)
    wind = (wspd / 10.0).where((wspd != MISSING) & wq.isin(GOOD_QUALITY))

    obs = pd.DataFrame({"temp_c": temp.to_numpy(), "wind_ms": wind.to_numpy()}, index=ts)
    obs = obs.dropna(how="all")
    return obs.resample("h").mean()


def load_hourly_weather(grid):
    """CISO hourly temperature and wind on the given UTC grid, equal-weighted across stations.

    Returns (temp, wind) Series on `grid`, with NaN where no station reported that hour.
    """
    years = sorted({t.year for t in grid})
    weights = pd.read_csv(WEIGHTS_FILE, comment="#")
    weights["station_id"] = weights["station_id"].astype(str)
    w_matrix = weights.pivot_table(index="station_id", columns="ba", values="weight", aggfunc="sum")

    per_var = {"temp_c": {}, "wind_ms": {}}
    for ghcn_id in w_matrix.index:
        isd_id = ISD_IDS[ghcn_id]
        frames = [parse_isd(fetch_isd_file(isd_id, y)) for y in years]
        st = pd.concat(frames).sort_index()
        st = st[~st.index.duplicated(keep="first")]
        for var in per_var:
            per_var[var][ghcn_id] = st[var]

    out = {}
    for var in per_var:
        wide = pd.DataFrame(per_var[var]).reindex(grid)
        out[var] = area_weighted_mean(wide, w_matrix)["CISO"].reindex(grid)
    return out["temp_c"], out["wind_ms"]
