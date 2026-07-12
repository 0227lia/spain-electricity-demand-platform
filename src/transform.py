from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from src.config import DAILY_DEMAND_PATH, MODEL_FEATURES_PATH, RAW_DATA_PATH


def add_spanish_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add known-in-advance national holiday and bridge-day indicators."""
    enriched = frame.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    enriched["year"] = enriched["date"].dt.year
    enriched["month"] = enriched["date"].dt.month
    enriched["week"] = enriched["date"].dt.isocalendar().week.astype(int)
    enriched["day_of_week"] = enriched["date"].dt.dayofweek
    enriched["is_weekend"] = enriched["day_of_week"].isin([5, 6])
    years = range(enriched["date"].dt.year.min(), enriched["date"].dt.year.max() + 1)
    holiday_calendar = holidays.country_holidays("ES", years=years, observed=True)
    dates = enriched["date"].dt.date
    enriched["is_holiday"] = dates.map(lambda value: value in holiday_calendar)
    enriched["is_day_before_holiday"] = dates.map(lambda value: value + timedelta(days=1) in holiday_calendar)
    enriched["is_day_after_holiday"] = dates.map(lambda value: value - timedelta(days=1) in holiday_calendar)
    previous_is_weekend = (enriched["date"] - pd.Timedelta(days=1)).dt.dayofweek.isin([5, 6])
    next_is_weekend = (enriched["date"] + pd.Timedelta(days=1)).dt.dayofweek.isin([5, 6])
    enriched["is_bridge_day"] = (
        ~enriched["is_holiday"]
        & ~enriched["is_weekend"]
        & (
            (enriched["is_day_before_holiday"] & previous_is_weekend)
            | (enriched["is_day_after_holiday"] & next_is_weekend)
        )
    )
    return enriched


def parse_ree_value(value: object) -> float:
    """Parse REE numeric strings that may use Spanish decimal separators."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Unsupported REE numeric value: {value!r}")

    cleaned = value.strip()
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", ".")
    return float(cleaned)


def validate_daily_demand(frame: pd.DataFrame) -> dict[str, int | float | str]:
    """Run explicit quality checks before loading data into the warehouse."""
    required = {"date", "demand_mwh"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required demand columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Demand dataset is empty")
    if frame["date"].isna().any() or frame["demand_mwh"].isna().any():
        raise ValueError("Demand dataset contains missing dates or values")
    if (frame["demand_mwh"] <= 0).any():
        raise ValueError("Demand values must be positive")
    duplicate_dates = int(frame["date"].duplicated().sum())
    if duplicate_dates:
        raise ValueError(f"Demand dataset has {duplicate_dates} duplicate dates")

    ordered = frame.sort_values("date")
    expected_dates = pd.date_range(ordered["date"].min(), ordered["date"].max(), freq="D")
    observed_dates = pd.DatetimeIndex(ordered["date"])
    missing_days = len(expected_dates.difference(observed_dates))
    if missing_days:
        raise ValueError(f"Demand dataset has {missing_days} missing daily observations")

    return {
        "row_count": int(len(frame)),
        "date_min": ordered["date"].min().date().isoformat(),
        "date_max": ordered["date"].max().date().isoformat(),
        "missing_days": int(missing_days),
        "duplicate_dates": duplicate_dates,
        "demand_min_mwh": float(ordered["demand_mwh"].min()),
        "demand_max_mwh": float(ordered["demand_mwh"].max()),
    }


def transform_raw_values(values: list[dict[str, object]]) -> pd.DataFrame:
    """Normalize the raw JSON records into a validated daily demand table."""
    frame = pd.DataFrame(values)
    if "datetime" not in frame.columns or "value" not in frame.columns:
        raise ValueError("Raw REE values must contain datetime and value")
    timestamps = pd.to_datetime(frame["datetime"], format="mixed", utc=True)
    demand = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps.dt.tz_convert("Europe/Madrid").dt.date),
            "demand_mwh": frame["value"].map(parse_ree_value),
        }
    ).sort_values("date")
    quality = validate_daily_demand(demand)
    demand["year"] = demand["date"].dt.year
    demand["month"] = demand["date"].dt.month
    demand["week"] = demand["date"].dt.isocalendar().week.astype(int)
    demand["day_of_week"] = demand["date"].dt.dayofweek
    demand["is_weekend"] = demand["day_of_week"].isin([5, 6])
    demand = add_spanish_calendar_features(demand)
    demand.attrs["quality"] = quality
    return demand.reset_index(drop=True)


def create_model_features(
    daily_demand: pd.DataFrame,
    national_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create known-calendar, exogenous-weather, and past-demand forecast features."""
    features = daily_demand.copy().sort_values("date")
    if "is_holiday" not in features:
        features = add_spanish_calendar_features(features)
    if national_weather is not None:
        weather = national_weather.copy()
        weather["date"] = pd.to_datetime(weather["date"])
        features = features.merge(weather, on="date", how="left", validate="one_to_one")
        weather_columns = [column for column in weather if column != "date"]
        if features[weather_columns].isna().any().any():
            raise ValueError("Weather proxy does not cover every demand date")
    day_of_year = features["date"].dt.dayofyear
    features["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    week_of_year = features["date"].dt.isocalendar().week.astype(int)
    features["week_of_year_sin"] = np.sin(2 * np.pi * week_of_year / 52.18)
    features["week_of_year_cos"] = np.cos(2 * np.pi * week_of_year / 52.18)
    for lag in [1, 7, 14, 28, 364]:
        features[f"lag_{lag}"] = features["demand_mwh"].shift(lag)
    shifted_demand = features["demand_mwh"].shift(1)
    for window in [7, 28, 56]:
        features[f"rolling_mean_{window}"] = shifted_demand.rolling(window).mean()
    for window in [7, 28]:
        features[f"rolling_std_{window}"] = shifted_demand.rolling(window).std()
    features["lag_1_minus_7"] = features["lag_1"] - features["lag_7"]
    return features.dropna().reset_index(drop=True)


def run_transform(
    raw_path: Path = RAW_DATA_PATH,
    national_weather: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float | str]]:
    """Read the raw snapshot, validate it, and create analytical/model-ready tables."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run `python -m src.extract` first.")
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    daily_demand = transform_raw_values(raw_payload["values"])
    model_features = create_model_features(daily_demand, national_weather)
    quality = daily_demand.attrs["quality"]
    DAILY_DEMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_demand.to_csv(DAILY_DEMAND_PATH, index=False)
    model_features.to_csv(MODEL_FEATURES_PATH, index=False)
    return daily_demand, model_features, quality


if __name__ == "__main__":
    daily, features, summary = run_transform()
    print(f"Validated {len(daily)} daily observations and generated {len(features)} model rows")
    print(summary)
