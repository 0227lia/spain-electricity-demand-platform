from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.config import SOURCE_NAME, WAREHOUSE_PATH


def build_warehouse(daily_demand: pd.DataFrame, warehouse_path: Path = WAREHOUSE_PATH) -> dict[str, int]:
    """Load validated demand data into a small DuckDB star schema."""
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    facts = daily_demand[["date", "demand_mwh"]].copy()
    facts["source_name"] = SOURCE_NAME
    dates = daily_demand[["date", "year", "month", "week", "day_of_week", "is_weekend"]].copy()

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
                CAST(is_weekend AS BOOLEAN) AS is_weekend
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
        fact_rows = int(connection.execute("SELECT COUNT(*) FROM fact_daily_demand").fetchone()[0])
        duplicate_dates = int(
            connection.execute("SELECT COUNT(*) - COUNT(DISTINCT date) FROM fact_daily_demand").fetchone()[0]
        )
        if duplicate_dates:
            raise ValueError(f"Warehouse load produced {duplicate_dates} duplicate fact dates")
        return {"fact_rows": fact_rows, "duplicate_dates": duplicate_dates}
    finally:
        connection.close()
