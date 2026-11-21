import numpy as np
import pandas as pd
import pytest

from pipeline.carbon_intensity import carbon_intensity
from pipeline.emissions import COMBUSTION_ONLY_G_PER_KWH, LIFECYCLE_AR5_G_PER_KWH

IDX = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")


def test_hand_computed_two_fuel_mix():
    # 100 MW coal + 300 MW wind -> (100*820 + 300*11) / 400 = 213.25
    gen = pd.DataFrame({"COL": [100.0], "WND": [300.0]}, index=IDX[:1])
    out = carbon_intensity(gen)
    assert out["ci_g_per_kwh"].iloc[0] == pytest.approx(213.25)
    assert out["mapped_share"].iloc[0] == pytest.approx(1.0)


def test_single_fuel_equals_its_factor():
    gen = pd.DataFrame({"NG": [50.0]}, index=IDX[:1])
    assert carbon_intensity(gen)["ci_g_per_kwh"].iloc[0] == pytest.approx(LIFECYCLE_AR5_G_PER_KWH["NG"])


def test_negative_generation_excluded_from_weights():
    # storage charging (-200 MW) must not pull CI down
    gen = pd.DataFrame({"COL": [100.0], "WAT": [-200.0]}, index=IDX[:1])
    out = carbon_intensity(gen)
    assert out["ci_g_per_kwh"].iloc[0] == pytest.approx(820.0)


def test_unmapped_fuel_reduces_coverage_not_ci_of_mapped_mix():
    # 100 MW coal + 100 MW OTH: CI uses coal only, coverage is 50%
    gen = pd.DataFrame({"COL": [100.0], "OTH": [100.0]}, index=IDX[:1])
    out = carbon_intensity(gen)
    assert out["ci_g_per_kwh"].iloc[0] == pytest.approx(820.0)
    assert out["mapped_share"].iloc[0] == pytest.approx(0.5)


def test_all_missing_or_zero_generation_gives_nan():
    gen = pd.DataFrame({"COL": [np.nan, 0.0], "WND": [np.nan, 0.0]}, index=IDX[:2])
    out = carbon_intensity(gen)
    assert out["ci_g_per_kwh"].isna().all()


def test_combustion_only_sensitivity_gives_zero_for_pure_renewables():
    gen = pd.DataFrame({"WND": [500.0], "SUN": [100.0]}, index=IDX[:1])
    out = carbon_intensity(gen, factors=COMBUSTION_ONLY_G_PER_KWH)
    assert out["ci_g_per_kwh"].iloc[0] == pytest.approx(0.0)
