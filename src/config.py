from __future__ import annotations

import os
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ree_demand_daily.json"
SOURCE_MANIFEST_PATH = DATA_DIR / "raw" / "source_manifest.json"
WEATHER_RAW_PATH = DATA_DIR / "raw" / "open_meteo_weather.json"
WEATHER_MANIFEST_PATH = DATA_DIR / "raw" / "weather_manifest.json"
PROCESSED_DIR = DATA_DIR / "processed"
DAILY_DEMAND_PATH = PROCESSED_DIR / "daily_demand.csv"
CITY_WEATHER_PATH = PROCESSED_DIR / "city_weather.csv"
NATIONAL_WEATHER_PATH = PROCESSED_DIR / "national_weather_proxy.csv"
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
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_DOCUMENTATION_URL = "https://open-meteo.com/en/docs/historical-weather-api"
WEATHER_SOURCE_NAME = "Open-Meteo Historical Weather API (ERA5)"
START_DATE = date.fromisoformat(os.getenv("REE_START_DATE", "2019-01-01"))
END_DATE = date.fromisoformat(os.getenv("REE_END_DATE", "2025-12-31"))
TEST_START_DATE = date.fromisoformat(os.getenv("FORECAST_TEST_START", "2025-01-01"))

RANDOM_STATE = 42
WEATHER_CITIES = {
    "madrid": {"latitude": 40.4168, "longitude": -3.7038, "weight": 0.2},
    "barcelona": {"latitude": 41.3874, "longitude": 2.1686, "weight": 0.2},
    "valencia": {"latitude": 39.4699, "longitude": -0.3763, "weight": 0.2},
    "sevilla": {"latitude": 37.3891, "longitude": -5.9845, "weight": 0.2},
    "bilbao": {"latitude": 43.2630, "longitude": -2.9350, "weight": 0.2},
}
WEATHER_DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
]
AUTOREGRESSIVE_FEATURES = [
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_bridge_day",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "day_of_year_sin",
    "day_of_year_cos",
    "week_of_year_sin",
    "week_of_year_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_364",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_std_7",
    "rolling_std_28",
    "lag_1_minus_7",
]
WEATHER_FEATURES = [
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "temperature_range_c",
    "city_temperature_spread_c",
    "heating_degree_days",
    "cooling_degree_days",
    "precipitation_mm",
    "wind_speed_max_kmh",
    "solar_radiation_mj_m2",
]
WEATHER_INFORMED_FEATURES = AUTOREGRESSIVE_FEATURES + WEATHER_FEATURES
BACKTEST_ORIGINS = [f"2024-{month:02d}-01" for month in range(1, 13)]
BACKTEST_HORIZON_DAYS = 28
