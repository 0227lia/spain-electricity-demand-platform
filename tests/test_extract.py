from __future__ import annotations

from datetime import date

from src.extract import build_request_url, extract_demand_values, yearly_windows


def test_yearly_windows_cover_cross_year_range() -> None:
    windows = yearly_windows(date(2023, 12, 30), date(2024, 1, 2))
    assert windows == [(date(2023, 12, 30), date(2023, 12, 31)), (date(2024, 1, 1), date(2024, 1, 2))]


def test_build_request_url_uses_daily_truncation() -> None:
    url = build_request_url(date(2024, 1, 1), date(2024, 1, 7))
    assert "time_trunc=day" in url
    assert "start_date=2024-01-01" in url


def test_extract_demand_values_selects_demand_indicator() -> None:
    payload = {
        "included": [
            {"attributes": {"title": "Other", "values": []}},
            {"attributes": {"title": "Demanda", "values": [{"datetime": "2024-01-01", "value": "1"}]}},
        ]
    }
    assert extract_demand_values(payload) == [{"datetime": "2024-01-01", "value": "1"}]
