from __future__ import annotations

import json
import logging

from src.analyze import run_sql_analytics
from src.config import PROJECT_ROOT, REPORTS_DIR
from src.extract import run_extraction
from src.forecast import run_forecast
from src.load import build_warehouse
from src.transform import run_transform


def configure_logging() -> None:
    logs_path = PROJECT_ROOT / "logs"
    logs_path.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(logs_path / "pipeline.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def run_pipeline() -> dict[str, object]:
    """Run extraction, validation, warehouse load, SQL analysis, and forecasting end to end."""
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting REE electricity-demand pipeline")
    extraction = run_extraction()
    daily_demand, model_features, data_quality = run_transform()
    warehouse_quality = build_warehouse(daily_demand)
    sql_reports = run_sql_analytics()
    forecast = run_forecast(model_features)
    summary = {
        "extraction": extraction,
        "data_quality": data_quality,
        "warehouse_quality": warehouse_quality,
        "sql_reports": sql_reports,
        "forecast": forecast,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Pipeline finished successfully")
    return summary


if __name__ == "__main__":
    result = run_pipeline()
    print(f"Processed {result['data_quality']['row_count']} daily demand records")
    print(f"Selected forecast model: {result['forecast']['model_selection']['selected_model']}")
