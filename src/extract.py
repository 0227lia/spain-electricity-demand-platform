from __future__ import annotations

import hashlib
import json
import logging
import ssl
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import truststore

from src.config import (
    API_BASE_URL,
    API_DOCUMENTATION_URL,
    END_DATE,
    RAW_DATA_PATH,
    SOURCE_MANIFEST_PATH,
    SOURCE_NAME,
    START_DATE,
)

LOGGER = logging.getLogger(__name__)


def yearly_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split an inclusive date range into API-friendly calendar-year windows."""
    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        window_end = min(date(cursor.year, 12, 31), end)
        windows.append((cursor, window_end))
        cursor = date(window_end.year + 1, 1, 1)
    return windows


def build_request_url(start: date, end: date) -> str:
    parameters = urlencode(
        {
            "start_date": f"{start.isoformat()}T00:00",
            "end_date": f"{end.isoformat()}T23:59",
            "time_trunc": "day",
        }
    )
    return f"{API_BASE_URL}?{parameters}"


def request_json(url: str, attempts: int = 3) -> dict[str, Any]:
    """Fetch JSON from REE with verified TLS and bounded retry attempts."""
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "portfolio-etl/1.0"})
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, context=context, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = exc
            if attempt < attempts:
                time.sleep(attempt)
    raise RuntimeError(f"REE API request failed after {attempts} attempts: {url}") from error


def extract_demand_values(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the daily national demand series from a REData response."""
    for indicator in payload.get("included", []):
        attributes = indicator.get("attributes", {})
        values = attributes.get("values")
        if attributes.get("title") == "Demanda" and isinstance(values, list):
            return values
    raise ValueError("REE response does not contain a daily 'Demanda' indicator")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_extraction(start: date = START_DATE, end: date = END_DATE) -> dict[str, Any]:
    """Download daily demand data and persist a consolidated raw snapshot plus provenance."""
    if start > end:
        raise ValueError("start date must not be after end date")

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_values: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for window_start, window_end in yearly_windows(start, end):
        url = build_request_url(window_start, window_end)
        payload = request_json(url)
        values = extract_demand_values(payload)
        all_values.extend(values)
        requests.append(
            {
                "url": url,
                "start_date": window_start.isoformat(),
                "end_date": window_end.isoformat(),
                "records": len(values),
            }
        )
        LOGGER.info("Downloaded %s daily records for %s", len(values), window_start.year)

    raw_snapshot = {
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "source_name": SOURCE_NAME,
        "source_documentation": API_DOCUMENTATION_URL,
        "requests": requests,
        "values": all_values,
    }
    RAW_DATA_PATH.write_text(json.dumps(raw_snapshot, indent=2), encoding="utf-8")
    manifest = {
        "retrieved_at_utc": raw_snapshot["retrieved_at_utc"],
        "source_name": SOURCE_NAME,
        "source_documentation": API_DOCUMENTATION_URL,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "request_count": len(requests),
        "records": len(all_values),
        "raw_file": RAW_DATA_PATH.name,
        "raw_file_sha256": sha256_file(RAW_DATA_PATH),
        "requests": requests,
    }
    SOURCE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_extraction()
    print(f"Downloaded {result['records']} daily demand records")
