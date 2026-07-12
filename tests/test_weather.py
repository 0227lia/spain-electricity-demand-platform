from __future__ import annotations

from datetime import date

import pytest

from src.config import WEATHER_CITIES, WEATHER_DAILY_VARIABLES
from src.weather import build_weather_url, transform_weather_payload


def sample_weather_payload() -> dict[str, object]:
    cities = []
    for index, (city, location) in enumerate(WEATHER_CITIES.items()):
        daily = {
            "time": ["2024-01-01", "2024-01-02"],
            "temperature_2m_mean": [10.0 + index, 11.0 + index],
            "temperature_2m_max": [15.0 + index, 16.0 + index],
            "temperature_2m_min": [5.0 + index, 6.0 + index],
            "precipitation_sum": [float(index), float(index + 1)],
            "wind_speed_10m_max": [20.0 + index, 21.0 + index],
            "shortwave_radiation_sum": [8.0 + index, 9.0 + index],
        }
        cities.append(
            {
                "city": city,
                "weight": location["weight"],
                "response": {"daily": daily},
            }
        )
    return {"cities": cities}


def test_weather_url_declares_era5_and_daily_variables() -> None:
    url = build_weather_url("madrid", date(2024, 1, 1), date(2024, 1, 2))
    assert "models=era5" in url
    assert "timezone=Europe%2FMadrid" in url
    for variable in WEATHER_DAILY_VARIABLES:
        assert variable in url


def test_weather_transform_builds_equal_weight_proxy() -> None:
    city, national, quality = transform_weather_payload(sample_weather_payload())
    assert len(city) == 10
    assert len(national) == 2
    assert national.iloc[0]["temperature_mean_c"] == pytest.approx(12.0)
    assert national.iloc[0]["city_temperature_spread_c"] == pytest.approx(4.0)
    assert national.iloc[0]["heating_degree_days"] == pytest.approx(6.0)
    assert quality["aggregation"] == "equal_weight"


def test_weather_transform_rejects_incomplete_city_coverage() -> None:
    payload = sample_weather_payload()
    payload["cities"] = payload["cities"][:-1]
    with pytest.raises(ValueError, match="every configured city"):
        transform_weather_payload(payload)
