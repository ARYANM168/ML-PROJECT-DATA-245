from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.capacity import SOURCE_TO_FUEL, capacity_by_year

RAW = Path("data/raw/eia930/eia930_CISO_fuel_type_2019-2025.csv")


def _capacity_csv(tmp_path):
    rows = [
        # period, BA, status, source, MW
        ("2024-12", "CISO", "OP", "SUN", 100.0),
        ("2024-12", "CISO", "OS", "SUN", 999.0),   # out of service: excluded
        ("2024-12", "CISO", "OP", "MWH", 50.0),    # storage: excluded from generation capacity
        ("2024-12", "CISO", "OP", "BIT", 20.0),    # coal maps to COL
        ("2024-12", "BANC", "OP", "SUN", 777.0),   # other balancing authority: excluded
        ("2024-12", "CISO", "OP", "XYZ", 5.0),     # unknown code maps to OTH
    ]
    df = pd.DataFrame(rows, columns=["period", "balancing_authority_code", "status", "energy_source_code", "nameplate-capacity-mw"])
    path = tmp_path / "cap.csv"
    df.to_csv(path, index=False)
    return path


def test_capacity_mapping_excludes_storage_status_and_other_ba(tmp_path):
    cap = capacity_by_year(_capacity_csv(tmp_path))
    assert cap.loc[2024, "SUN"] == pytest.approx(100.0)
    assert cap.loc[2024, "COL"] == pytest.approx(20.0)
    assert cap.loc[2024, "OTH"] == pytest.approx(5.0)
    assert "MWH" not in SOURCE_TO_FUEL


@pytest.mark.skipif(not RAW.exists(), reason="raw EIA-930 data not present")
def test_smoke_build_one_month_horizon_1():
    from pipeline.run_pipeline import build_panel

    panel, feature_cols, record, *_ = build_panel("2024-10-01", "2024-10-31", horizon=1)
    assert len(feature_cols) == 28  # 24 CI/demand/calendar + 4 hourly weather
    for col in ["wx_temp_lag1", "wx_wind_roll24_mean", "wx_any_imputed", "local_time", "dst_flag", "ci_mapped_share", "ci_hydro_missing",
                "outlier_negative_sun", "outlier_above_nameplate_sun", "target_ci_lifecycle", "split"]:
        assert col in panel.columns, col
    assert panel["dst_flag"].dtype == bool
    share = panel["ci_mapped_share"].dropna()
    assert ((share >= 0) & (share <= 1)).all()
    assert record["rows_after_sparse_day_rule"] == len(panel)
    assert "outlier_above_nameplate_oth" not in panel.columns  # OTH has no nameplate check
