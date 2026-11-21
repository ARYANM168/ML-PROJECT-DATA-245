"""Emission factors in gCO2eq per kWh, keyed by EIA-930 fuel code.

LIFECYCLE_AR5_G_PER_KWH uses IPCC AR5 (2014, Annex III) lifecycle medians.
COMBUSTION_ONLY_G_PER_KWH is the sensitivity run: direct stack CO2 only.

VERIFY every value against the published tables before the proposal or the
report cites it. The mapping below is a working draft, not a sourced table.
"""

LIFECYCLE_AR5_G_PER_KWH = {
    "COL": 820.0,  # coal (pulverized)
    "NG": 490.0,   # gas, combined cycle median; EIA-930 does not split peakers
    # OIL is not in the IPCC AR5 Table A.III.2 medians, so it has no lifecycle
    # factor here. It is excluded from CI and reported as unmapped.
    "NUC": 12.0,
    "WAT": 24.0,   # hydropower
    "WND": 11.0,   # onshore median; EIA-930 does not split offshore wind
    "SUN": 48.0,   # utility-scale PV
    "GEO": 38.0,
    "BIO": 230.0,  # dedicated biomass median
}

# Assumed heat rates (MMBtu/MWh) x IPCC 2006 default CO2 per MMBtu, for
# fossil fuels. Renewables, nuclear, and biomass count as zero (biogenic
# CO2 is excluded in the combustion-only view). These heat rates are
# assumptions for the sensitivity run and must be stated in the report.
COMBUSTION_ONLY_G_PER_KWH = {
    "COL": 993.0,  # 94.6 kg/MMBtu x 10.5
    "NG": 398.0,   # 53.06 kg/MMBtu x 7.5
    "OIL": 778.0,  # 74.1 kg/MMBtu x 10.5
    "NUC": 0.0,
    "WAT": 0.0,
    "WND": 0.0,
    "SUN": 0.0,
    "GEO": 0.0,
    "BIO": 0.0,
}

# Codes with no defensible factor. They are left out of CI and reported.
UNMAPPED_FUELS = ("OTH", "UNK")
