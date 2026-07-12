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


def test_build_warehouse_loads_weather_mart(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    daily = pd.DataFrame(
        {
            "date": dates,
            "demand_mwh": [100.0, 110.0, 120.0],
            "year": [2024] * 3,
            "month": [1] * 3,
            "week": [1] * 3,
            "day_of_week": [0, 1, 2],
            "is_weekend": [False] * 3,
            "is_holiday": [True, False, False],
            "is_bridge_day": [False, True, False],
        }
    )
    city_weather = pd.DataFrame(
        {
            "date": dates,
            "city": ["madrid"] * 3,
            "weight": [1.0] * 3,
            "temperature_mean_c": [8.0, 9.0, 10.0],
        }
    )
    national_weather = pd.DataFrame(
        {
            "date": dates,
            "temperature_mean_c": [8.0, 9.0, 10.0],
            "heating_degree_days": [10.0, 9.0, 8.0],
            "cooling_degree_days": [0.0] * 3,
            "precipitation_mm": [0.0, 1.0, 0.0],
            "wind_speed_max_kmh": [10.0, 11.0, 12.0],
        }
    )
    warehouse = tmp_path / "electricity.duckdb"
    result = build_warehouse(
        daily,
        warehouse,
        city_weather=city_weather,
        national_weather=national_weather,
    )
    connection = duckdb.connect(str(warehouse), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM mart_daily_demand_weather").fetchone()[0] == 3
    finally:
        connection.close()
    assert result["weather_city_rows"] == 3
    assert result["weather_proxy_rows"] == 3
