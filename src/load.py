from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.config import SOURCE_NAME, WAREHOUSE_PATH, WEATHER_SOURCE_NAME


def build_warehouse(
    daily_demand: pd.DataFrame,
    warehouse_path: Path = WAREHOUSE_PATH,
    city_weather: pd.DataFrame | None = None,
    national_weather: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Load validated demand data into a small DuckDB star schema."""
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    facts = daily_demand[["date", "demand_mwh"]].copy()
    facts["source_name"] = SOURCE_NAME
    date_columns = [
        "date",
        "year",
        "month",
        "week",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "is_bridge_day",
    ]
    dates = daily_demand.copy()
    for column in ["is_holiday", "is_bridge_day"]:
        if column not in dates:
            dates[column] = False
    dates = dates[date_columns]

    connection = duckdb.connect(str(warehouse_path))
    try:
        connection.register("facts_frame", facts)
        connection.register("dates_frame", dates)
        connection.execute(
            """
            CREATE OR REPLACE TABLE fact_daily_demand AS
            SELECT CAST(date AS DATE) AS date, CAST(demand_mwh AS DOUBLE) AS demand_mwh, source_name
            FROM facts_frame
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE dim_date AS
            SELECT
                CAST(date AS DATE) AS date,
                CAST(year AS INTEGER) AS year,
                CAST(month AS INTEGER) AS month,
                CAST(week AS INTEGER) AS week,
                CAST(day_of_week AS INTEGER) AS day_of_week,
                CAST(is_weekend AS BOOLEAN) AS is_weekend,
                CAST(is_holiday AS BOOLEAN) AS is_holiday,
                CAST(is_bridge_day AS BOOLEAN) AS is_bridge_day
            FROM dates_frame
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW mart_monthly_demand AS
            SELECT
                date_trunc('month', fact.date) AS month,
                SUM(fact.demand_mwh) AS total_demand_mwh,
                AVG(fact.demand_mwh) AS avg_daily_demand_mwh,
                MAX(fact.demand_mwh) AS max_daily_demand_mwh
            FROM fact_daily_demand AS fact
            GROUP BY 1
            """
        )
        if city_weather is not None and national_weather is not None:
            city_frame = city_weather.copy()
            city_frame["source_name"] = WEATHER_SOURCE_NAME
            national_frame = national_weather.copy()
            national_frame["source_name"] = WEATHER_SOURCE_NAME
            connection.register("city_weather_frame", city_frame)
            connection.register("national_weather_frame", national_frame)
            connection.execute(
                """
                CREATE OR REPLACE TABLE fact_daily_weather_city AS
                SELECT * REPLACE (CAST(date AS DATE) AS date)
                FROM city_weather_frame
                """
            )
            connection.execute(
                """
                CREATE OR REPLACE TABLE fact_daily_weather_proxy AS
                SELECT * REPLACE (CAST(date AS DATE) AS date)
                FROM national_weather_frame
                """
            )
            connection.execute(
                """
                CREATE OR REPLACE VIEW mart_daily_demand_weather AS
                SELECT
                    demand.date,
                    demand.demand_mwh,
                    weather.temperature_mean_c,
                    weather.heating_degree_days,
                    weather.cooling_degree_days,
                    weather.precipitation_mm,
                    weather.wind_speed_max_kmh,
                    calendar.is_holiday,
                    calendar.is_bridge_day
                FROM fact_daily_demand AS demand
                JOIN fact_daily_weather_proxy AS weather USING (date)
                JOIN dim_date AS calendar USING (date)
                """
            )
        fact_rows = int(connection.execute("SELECT COUNT(*) FROM fact_daily_demand").fetchone()[0])
        duplicate_dates = int(
            connection.execute("SELECT COUNT(*) - COUNT(DISTINCT date) FROM fact_daily_demand").fetchone()[0]
        )
        if duplicate_dates:
            raise ValueError(f"Warehouse load produced {duplicate_dates} duplicate fact dates")
        quality = {"fact_rows": fact_rows, "duplicate_dates": duplicate_dates}
        if city_weather is not None and national_weather is not None:
            quality["weather_city_rows"] = int(
                connection.execute("SELECT COUNT(*) FROM fact_daily_weather_city").fetchone()[0]
            )
            quality["weather_proxy_rows"] = int(
                connection.execute("SELECT COUNT(*) FROM fact_daily_weather_proxy").fetchone()[0]
            )
        return quality
    finally:
        connection.close()
