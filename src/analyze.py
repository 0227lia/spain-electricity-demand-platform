from __future__ import annotations

from pathlib import Path

import duckdb

from src.config import SQL_PATH, SQL_REPORTS_DIR, WAREHOUSE_PATH


def parse_named_queries(sql_path: Path = SQL_PATH) -> dict[str, str]:
    """Read named SQL blocks delimited by `-- name: query_name` comments."""
    queries: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in sql_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-- name:"):
            if current_name:
                queries[current_name] = "\n".join(current_lines).strip()
            current_name = line.split(":", 1)[1].strip()
            current_lines = []
        elif current_name:
            current_lines.append(line)
    if current_name:
        queries[current_name] = "\n".join(current_lines).strip()
    if not queries:
        raise ValueError(f"No named SQL queries found in {sql_path}")
    return queries


def run_sql_analytics(warehouse_path: Path = WAREHOUSE_PATH, sql_path: Path = SQL_PATH) -> dict[str, int]:
    """Execute analytical SQL queries and export every result as a CSV report."""
    SQL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = parse_named_queries(sql_path)
    connection = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        row_counts: dict[str, int] = {}
        for name, query in queries.items():
            result = connection.execute(query).df()
            result.to_csv(SQL_REPORTS_DIR / f"{name}.csv", index=False)
            row_counts[name] = len(result)
        return row_counts
    finally:
        connection.close()
