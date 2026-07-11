from __future__ import annotations

import os
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ree_demand_daily.json"
SOURCE_MANIFEST_PATH = DATA_DIR / "raw" / "source_manifest.json"
PROCESSED_DIR = DATA_DIR / "processed"
DAILY_DEMAND_PATH = PROCESSED_DIR / "daily_demand.csv"
MODEL_FEATURES_PATH = PROCESSED_DIR / "model_features.csv"
WAREHOUSE_PATH = PROJECT_ROOT / "warehouse" / "electricity.duckdb"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SQL_REPORTS_DIR = REPORTS_DIR / "sql"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "demand_hgb.joblib"
SQL_PATH = PROJECT_ROOT / "sql" / "analytics.sql"

API_BASE_URL = "https://apidatos.ree.es/es/datos/demanda/evolucion"
API_DOCUMENTATION_URL = "https://www.ree.es/en/datos/apidata"
SOURCE_NAME = "Red Electrica de Espana REData API"
START_DATE = date.fromisoformat(os.getenv("REE_START_DATE", "2019-01-01"))
END_DATE = date.fromisoformat(os.getenv("REE_END_DATE", "2025-12-31"))
TEST_START_DATE = date.fromisoformat(os.getenv("FORECAST_TEST_START", "2025-01-01"))

RANDOM_STATE = 42
FORECAST_FEATURES = [
    "day_of_week",
    "month",
    "is_weekend",
    "day_of_year_sin",
    "day_of_year_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]
BACKTEST_ORIGINS = ["2024-03-01", "2024-06-01", "2024-09-01", "2024-12-01"]
BACKTEST_HORIZON_DAYS = 28
