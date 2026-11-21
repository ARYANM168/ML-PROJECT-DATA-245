"""
config.py - shared settings for Part 3 (day-type clustering and scheduling benefit).

Part 3 reuses the same hourly CAISO carbon-intensity target that Part 1
(data pipeline) builds and Part 2 (forecasting) caches as caiso_hourly.csv.
All paths are relative to the project folder so the scripts run anywhere.
"""

from pathlib import Path

# ----------------------------- paths -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "caiso_hourly.csv"   # timestamp, carbon_intensity
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

TIME_COL = "timestamp"
TARGET_COL = "carbon_intensity"          # gCO2e per kWh (direct combustion factors)

# ----------------------------- cleaning -----------------------------
MAX_INTERP_HOURS = 2       # fill gaps of at most this many hours inside a day
MIN_VALID_CI = 1.0         # values below this are treated as reporting errors

# ----------------------------- leakage-safe split -----------------------------
# Clusters, typical-hour tables and the day-type classifier are fitted only on
# the training period. Every scheduling result is reported on the test period.
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
# EIA-930 changed its energy-source categories from 2024 Q3 (battery / other /
# unknown energy storage etc.).  Results are also reported on the clean window
# before the change so they do not depend on how those new labels are mapped.
CLEAN_TEST_END = "2024-06-30"

# ----------------------------- clustering -----------------------------
K_RANGE = range(2, 11)     # candidate numbers of day types
K_FINAL = None             # None = choose k automatically (see clustering.py)
RANDOM_STATE = 42
N_INIT = 50

# ----------------------------- scheduling simulation -----------------------------
FLEX_ENERGY_MWH = 1.0      # flexible energy per day (results scale linearly)
BLOCK_HOURS = 4            # main scenario: one contiguous 4-hour block per day
BLOCK_SENSITIVITY = [1, 2, 3, 4, 6, 8]
EVENING_START = 18         # "business-as-usual" evening block (18:00-22:00)
# Fleet example used to translate per-MWh savings into tonnes per year
FLEET_FLEX_MWH_PER_DAY = 1000.0   # e.g. ~100,000 EVs x 10 kWh each
