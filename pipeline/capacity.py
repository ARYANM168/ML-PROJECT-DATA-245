"""Operating generation capacity by EIA-930 fuel code, from the December EIA-860 snapshots.

Limitations (state them in the report):
- The snapshot is California only (EIA's balancing-authority facet is not usable on this
  route). CISO's few out-of-state resources are therefore missing, so capacity is a slight
  underestimate for CISO.
- One December snapshot per year is used as that year's nameplate. Capacity changes within
  a year are not captured.
- Storage (MWH) is excluded from generation capacity. Only status OP (operating) counts.
"""

from pathlib import Path

import pandas as pd

CAPACITY_FILE = Path("data/raw/eia860/eia_capacity_CA_december_2019-2025.csv")
BA = "CISO"
OPERATING = {"OP"}

# EIA-860 energy_source_code -> EIA-930 fuel code. Codes not listed map to OTH.
SOURCE_TO_FUEL = {
    "BIT": "COL", "RC": "COL", "SUB": "COL", "LIG": "COL", "WC": "COL", "ANT": "COL",
    "NG": "NG",
    "DFO": "OIL", "RFO": "OIL", "KER": "OIL", "JF": "OIL", "WO": "OIL", "PC": "OIL", "PG": "OIL",
    "NUC": "NUC",
    "WAT": "WAT",
    "WND": "WND",
    "SUN": "SUN",
    "GEO": "GEO",
    "WDS": "BIO", "OBS": "BIO", "AB": "BIO", "OBG": "BIO", "OBL": "BIO", "BLQ": "BIO",
    "LFG": "BIO", "WDL": "BIO",
}
STORAGE_CODES = {"MWH"}


def capacity_by_year(path=CAPACITY_FILE):
    """Return DataFrame: index = year, columns = EIA-930 fuel codes, values = MW (operating only)."""
    c = pd.read_csv(path)
    c = c[(c["balancing_authority_code"] == BA) & (c["status"].isin(OPERATING))]
    c = c[~c["energy_source_code"].isin(STORAGE_CODES)].copy()
    c["fuel"] = c["energy_source_code"].map(SOURCE_TO_FUEL).fillna("OTH")
    c["year"] = c["period"].str.slice(0, 4).astype(int)
    out = c.pivot_table(index="year", columns="fuel", values="nameplate-capacity-mw", aggfunc="sum", fill_value=0.0)
    return out
