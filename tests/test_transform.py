from __future__ import annotations

import pandas as pd
import pytest

from src.transform import create_model_features, parse_ree_value, transform_raw_values, validate_daily_demand


def make_daily_frame(rows: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame({"date": dates, "demand_mwh": range(1, rows + 1)})


def test_parse_ree_value_handles_spanish_and_plain_decimals() -> None:
    assert parse_ree_value("555244,867") == pytest.approx(555244.867)
    assert parse_ree_value("1.234,5") == pytest.approx(1234.5)
    assert parse_ree_value(12.5) == pytest.approx(12.5)


def test_validation_rejects_missing_calendar_day() -> None:
    frame = make_daily_frame(4).drop(index=2)
    with pytest.raises(ValueError, match="missing daily observations"):
        validate_daily_demand(frame)


def test_model_features_use_only_past_values() -> None:
    features = create_model_features(make_daily_frame(50))
    first_row = features.iloc[0]
    assert first_row["demand_mwh"] == 29
    assert first_row["lag_1"] == 28
    assert first_row["lag_7"] == 22
    assert first_row["rolling_mean_7"] == pytest.approx(25.0)


def test_transform_raw_values_normalizes_dates_and_values() -> None:
    values = [
        {"datetime": "2024-01-01T00:00:00.000+01:00", "value": "100,5"},
        {"datetime": "2024-01-02T00:00:00.000+01:00", "value": "101,5"},
    ]
    frame = transform_raw_values(values)
    assert list(frame["demand_mwh"]) == [100.5, 101.5]
    assert list(frame["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02"]
