from __future__ import annotations

import duckdb
import pandas as pd

from src.load import build_warehouse


def test_build_warehouse_creates_fact_table(tmp_path) -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "demand_mwh": [100.0, 110.0, 120.0],
            "year": [2024, 2024, 2024],
            "month": [1, 1, 1],
            "week": [1, 1, 1],
            "day_of_week": [0, 1, 2],
            "is_weekend": [False, False, False],
        }
    )
    warehouse = tmp_path / "electricity.duckdb"
    result = build_warehouse(daily, warehouse)
    connection = duckdb.connect(str(warehouse), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM fact_daily_demand").fetchone()[0] == 3
    finally:
        connection.close()
    assert result == {"fact_rows": 3, "duplicate_dates": 0}
