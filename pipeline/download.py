"""Raw data downloads: EIA-930 (hourly demand and fuel-type generation) and NOAA CDO (daily GHCND).

Keys are read from environment variables (EIA_API_KEY, NOAA_CDO_TOKEN). Load them
from .env first with: set -a; . ./.env; set +a

Every downloaded file is checksummed, logged to data/raw/manifest.json, and made
read-only through pipeline.provenance.record_download. The source URLs stored in
the manifest never include the API key.
"""

import csv
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from .provenance import record_download

RAW_DIR = Path("data/raw")
MANIFEST = RAW_DIR / "manifest.json"
EIA_BASE = "https://api.eia.gov/v2/electricity/rto"
NOAA_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
PAGE_ROWS = 5000  # EIA v2 maximum rows per request
NOAA_PAGE_ROWS = 1000  # CDO maximum rows per request
PUDL_NOTE = "2025.12.1 (installed in pudl env; not used by this download)"


def _get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _write_csv(path, rows, fieldnames):
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Raw files are read-only and are never overwritten; "
            "delete it deliberately and update the manifest if a re-pull is needed."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fetch_eia_region(route, respondent, start, end, out_name):
    """Pull all hourly rows for one respondent from an EIA-930 route.

    route : "region-data" (demand, generation, interchange) or "fuel-type-data".
    start, end : "YYYY-MM-DDTHH" strings in UTC.
    Writes RAW_DIR/eia930/<out_name>.csv and logs the file to the manifest.
    """
    api_key = os.environ["EIA_API_KEY"]
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        params = [
            ("api_key", api_key),
            ("frequency", "hourly"),
            ("data[0]", "value"),
            ("facets[respondent][]", respondent),
            ("start", start),
            ("end", end),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"),
            ("offset", str(offset)),
            ("length", str(PAGE_ROWS)),
        ]
        page = _get_json(f"{EIA_BASE}/{route}/data/?{urllib.parse.urlencode(params)}")
        total = int(page["response"]["total"])
        data = page["response"]["data"]
        if not data:
            break
        rows.extend(data)
        offset += len(data)

    out = RAW_DIR / "eia930" / f"{out_name}.csv"
    fields = sorted({k for r in rows for k in r.keys()})
    _write_csv(out, rows, fields)

    public_url = f"{EIA_BASE}/{route}/data/?respondent={respondent}&frequency=hourly&start={start}&end={end}"
    record_download(out, public_url, pudl_version=PUDL_NOTE, manifest_path=MANIFEST)
    return out, len(rows)


def fetch_noaa_daily(station_id, datatypes, start_year, end_year, out_name):
    """Pull GHCND daily values for one station, one calendar year at a time.

    station_id : "USW00023174" (without the GHCND: prefix).
    datatypes : list such as ["TMAX", "TMIN", "TAVG", "AWND"].
    CDO limits daily queries to one year per request, so this loops by year.
    """
    token = os.environ["NOAA_CDO_TOKEN"]
    headers = {"token": token}
    rows = []
    for year in range(start_year, end_year + 1):
        offset = 1
        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": f"GHCND:{station_id}",
                "datatypeid": ",".join(datatypes),
                "startdate": f"{year}-01-01",
                "enddate": f"{year}-12-31",
                "units": "metric",
                "limit": NOAA_PAGE_ROWS,
                "offset": offset,
            }
            page = _get_json(f"{NOAA_BASE}/data?{urllib.parse.urlencode(params)}", headers=headers)
            results = page.get("results", [])
            rows.extend(results)
            count = page.get("metadata", {}).get("resultset", {}).get("count", 0)
            offset += NOAA_PAGE_ROWS
            if offset > count:
                break

    out = RAW_DIR / "noaa" / f"{out_name}.csv"
    _write_csv(out, rows, ["date", "datatype", "station", "attributes", "value"])

    public_url = (
        f"{NOAA_BASE}/data?datasetid=GHCND&stationid=GHCND:{station_id}"
        f"&datatypeid={','.join(datatypes)}&startdate={start_year}-01-01&enddate={end_year}-12-31&units=metric"
    )
    record_download(out, public_url, pudl_version=PUDL_NOTE, manifest_path=MANIFEST)
    return out, len(rows)


def fetch_capacity_snapshots(years, out_name="eia_capacity_CA_december_2019-2025"):
    """December generator-capacity snapshots for California from EIA-860 monthly data.

    One snapshot per year. Filtering to CISO happens in pipeline/capacity.py, since the API
    facet on balancing authority is not accepted for this route.
    """
    api_key = os.environ["EIA_API_KEY"]
    url_base = "https://api.eia.gov/v2/electricity/operating-generator-capacity/data/"
    rows = []
    for year in years:
        offset, total = 0, None
        while total is None or offset < total:
            params = [
                ("api_key", api_key),
                ("frequency", "monthly"),
                ("data[0]", "nameplate-capacity-mw"),
                ("facets[stateid][]", "CA"),
                ("start", f"{year}-12"),
                ("end", f"{year}-12"),
                ("sort[0][column]", "period"),
                ("sort[0][direction]", "desc"),
                ("offset", str(offset)),
                ("length", "5000"),
            ]
            page = _get_json(f"{url_base}?{urllib.parse.urlencode(params)}")
            total = int(page["response"]["total"])
            data = page["response"]["data"]
            if not data:
                break
            rows.extend(data)
            offset += len(data)

    out = RAW_DIR / "eia860" / f"{out_name}.csv"
    fields = sorted({k for r in rows for k in r.keys()})
    _write_csv(out, rows, fields)
    public_url = f"{url_base}?frequency=monthly&facets[stateid][]=CA&start=YYYY-12&end=YYYY-12"
    record_download(out, public_url, pudl_version=PUDL_NOTE, manifest_path=MANIFEST)
    return out, len(rows)
