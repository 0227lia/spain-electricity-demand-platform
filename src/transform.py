from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DAILY_DEMAND_PATH, MODEL_FEATURES_PATH, RAW_DATA_PATH


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
    demand.attrs["quality"] = quality
    return demand.reset_index(drop=True)


def create_model_features(daily_demand: pd.DataFrame) -> pd.DataFrame:
    """Create one-step-ahead features using only demand observed before the target day."""
    features = daily_demand.copy().sort_values("date")
    day_of_year = features["date"].dt.dayofyear
    features["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    for lag in [1, 7, 14, 28]:
        features[f"lag_{lag}"] = features["demand_mwh"].shift(lag)
    features["rolling_mean_7"] = features["demand_mwh"].shift(1).rolling(7).mean()
    features["rolling_mean_28"] = features["demand_mwh"].shift(1).rolling(28).mean()
    return features.dropna().reset_index(drop=True)


def run_transform(
    raw_path: Path = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float | str]]:
    """Read the raw snapshot, validate it, and create analytical/model-ready tables."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Run `python -m src.extract` first.")
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    daily_demand = transform_raw_values(raw_payload["values"])
    model_features = create_model_features(daily_demand)
    quality = daily_demand.attrs["quality"]
    DAILY_DEMAND_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_demand.to_csv(DAILY_DEMAND_PATH, index=False)
    model_features.to_csv(MODEL_FEATURES_PATH, index=False)
    return daily_demand, model_features, quality


if __name__ == "__main__":
    daily, features, summary = run_transform()
    print(f"Validated {len(daily)} daily observations and generated {len(features)} model rows")
    print(summary)
