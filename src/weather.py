from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from src.config import (
    CITY_WEATHER_PATH,
    END_DATE,
    NATIONAL_WEATHER_PATH,
    START_DATE,
    WEATHER_API_URL,
    WEATHER_CITIES,
    WEATHER_DAILY_VARIABLES,
    WEATHER_DOCUMENTATION_URL,
    WEATHER_MANIFEST_PATH,
    WEATHER_RAW_PATH,
    WEATHER_SOURCE_NAME,
)
from src.extract import request_json

WEATHER_COLUMN_NAMES = {
    "temperature_2m_mean": "temperature_mean_c",
    "temperature_2m_max": "temperature_max_c",
    "temperature_2m_min": "temperature_min_c",
    "precipitation_sum": "precipitation_mm",
    "wind_speed_10m_max": "wind_speed_max_kmh",
    "shortwave_radiation_sum": "solar_radiation_mj_m2",
}


def build_weather_url(city: str, start: date, end: date) -> str:
    """Build a deterministic ERA5 daily request for one configured city."""
    if city not in WEATHER_CITIES:
        raise ValueError(f"Unknown weather city: {city}")
    location = WEATHER_CITIES[city]
    parameters = urlencode(
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": ",".join(WEATHER_DAILY_VARIABLES),
            "timezone": "Europe/Madrid",
            "models": "era5",
        }
    )
    return f"{WEATHER_API_URL}?{parameters}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_weather_extraction(start: date = START_DATE, end: date = END_DATE) -> dict[str, Any]:
    """Download one ERA5 daily snapshot per city and persist its provenance."""
    if start > end:
        raise ValueError("start date must not be after end date")

    retrieved_at = datetime.now(UTC).isoformat()
    cities: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for city, location in WEATHER_CITIES.items():
        url = build_weather_url(city, start, end)
        payload = request_json(url)
        daily = payload.get("daily", {})
        dates = daily.get("time", [])
        if not isinstance(dates, list) or not dates:
            raise ValueError(f"Open-Meteo returned no daily weather for {city}")
        cities.append(
            {
                "city": city,
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "weight": location["weight"],
                "response": payload,
            }
        )
        requests.append({"city": city, "url": url, "records": len(dates)})

    snapshot = {
        "retrieved_at_utc": retrieved_at,
        "source_name": WEATHER_SOURCE_NAME,
        "source_documentation": WEATHER_DOCUMENTATION_URL,
        "aggregation": "Equal-weight proxy across five configured mainland cities.",
        "cities": cities,
    }
    WEATHER_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEATHER_RAW_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    manifest = {
        "retrieved_at_utc": retrieved_at,
        "source_name": WEATHER_SOURCE_NAME,
        "source_documentation": WEATHER_DOCUMENTATION_URL,
        "model": "ERA5",
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "city_count": len(cities),
        "records": sum(item["records"] for item in requests),
        "raw_file": WEATHER_RAW_PATH.name,
        "raw_file_sha256": _sha256(WEATHER_RAW_PATH),
        "requests": requests,
    }
    WEATHER_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def transform_weather_payload(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Normalize city weather and create an equal-weight national weather proxy."""
    city_frames: list[pd.DataFrame] = []
    for city_payload in payload.get("cities", []):
        city = str(city_payload["city"])
        daily = city_payload.get("response", {}).get("daily", {})
        required = {"time", *WEATHER_COLUMN_NAMES}
        missing = sorted(required - set(daily))
        if missing:
            raise ValueError(f"Weather response for {city} is missing: {missing}")
        lengths = {len(daily[column]) for column in required}
        if len(lengths) != 1:
            raise ValueError(f"Weather response for {city} has inconsistent column lengths")
        frame = pd.DataFrame({column: daily[column] for column in required})
        frame = frame.rename(columns={"time": "date", **WEATHER_COLUMN_NAMES})
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["city"] = city
        frame["weight"] = float(city_payload["weight"])
        city_frames.append(frame)

    if not city_frames:
        raise ValueError("Weather snapshot contains no city responses")
    city_weather = pd.concat(city_frames, ignore_index=True).sort_values(["date", "city"])
    numeric_columns = list(WEATHER_COLUMN_NAMES.values())
    city_weather[numeric_columns] = city_weather[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if city_weather[numeric_columns].isna().any().any():
        raise ValueError("Weather snapshot contains missing or non-numeric daily values")
    if city_weather.duplicated(["date", "city"]).any():
        raise ValueError("Weather snapshot contains duplicate city-date rows")

    expected_city_count = len(WEATHER_CITIES)
    coverage = city_weather.groupby("date").agg(
        city_count=("city", "nunique"),
        weight_sum=("weight", "sum"),
    )
    if not coverage["city_count"].eq(expected_city_count).all():
        raise ValueError("Weather dates do not contain every configured city")
    if not np.allclose(coverage["weight_sum"], 1.0):
        raise ValueError("Weather city weights do not sum to one for every date")

    weighted = city_weather[["date", "weight", *numeric_columns]].copy()
    for column in numeric_columns:
        weighted[column] = weighted[column] * weighted["weight"]
    national = weighted.groupby("date", as_index=False)[numeric_columns].sum()
    temperature_spread = city_weather.groupby("date")["temperature_mean_c"].agg(
        lambda values: values.max() - values.min()
    )
    national["city_temperature_spread_c"] = national["date"].map(temperature_spread)
    national["temperature_range_c"] = national["temperature_max_c"] - national["temperature_min_c"]
    national["heating_degree_days"] = (18.0 - national["temperature_mean_c"]).clip(lower=0)
    national["cooling_degree_days"] = (national["temperature_mean_c"] - 22.0).clip(lower=0)
    national = national.sort_values("date").reset_index(drop=True)

    expected_dates = pd.date_range(national["date"].min(), national["date"].max(), freq="D")
    missing_days = len(expected_dates.difference(pd.DatetimeIndex(national["date"])))
    if missing_days:
        raise ValueError(f"Weather proxy has {missing_days} missing daily observations")
    quality = {
        "city_rows": int(len(city_weather)),
        "national_rows": int(len(national)),
        "city_count": expected_city_count,
        "date_min": national["date"].min().date().isoformat(),
        "date_max": national["date"].max().date().isoformat(),
        "missing_days": int(missing_days),
        "aggregation": "equal_weight",
    }
    return city_weather.reset_index(drop=True), national, quality


def run_weather_transform(
    raw_path: Path = WEATHER_RAW_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Transform the persisted Open-Meteo snapshot into two validated tables."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Weather data not found at {raw_path}. Run weather extraction first.")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    city_weather, national_weather, quality = transform_weather_payload(payload)
    CITY_WEATHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    city_weather.to_csv(CITY_WEATHER_PATH, index=False)
    national_weather.to_csv(NATIONAL_WEATHER_PATH, index=False)
    return city_weather, national_weather, quality


if __name__ == "__main__":
    extraction = run_weather_extraction()
    _, national, quality_summary = run_weather_transform()
    print(f"Downloaded {extraction['records']} city-day weather records")
    print(f"Built {len(national)} national-proxy daily rows: {quality_summary}")
