from __future__ import annotations

import numpy as np

from src.forecast import regression_metrics, smape


def test_smape_is_zero_for_perfect_forecast() -> None:
    assert smape(np.array([10.0, 20.0]), np.array([10.0, 20.0])) == 0.0


def test_regression_metrics_include_expected_fields() -> None:
    metrics = regression_metrics(np.array([10.0, 20.0]), np.array([11.0, 18.0]))
    assert set(metrics) == {"mae_mwh", "rmse_mwh", "smape_pct"}
    assert metrics["mae_mwh"] == 1.5
