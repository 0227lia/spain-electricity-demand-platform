from __future__ import annotations

import numpy as np

from src.forecast import (
    conformal_radius,
    interval_diagnostics,
    moving_block_bootstrap_improvement,
    regression_metrics,
    smape,
)


def test_smape_is_zero_for_perfect_forecast() -> None:
    assert smape(np.array([10.0, 20.0]), np.array([10.0, 20.0])) == 0.0


def test_regression_metrics_include_expected_fields() -> None:
    metrics = regression_metrics(np.array([10.0, 20.0]), np.array([11.0, 18.0]))
    assert set(metrics) == {
        "mae_mwh",
        "rmse_mwh",
        "smape_pct",
        "wape_pct",
        "bias_mwh",
        "mase_7d",
    }
    assert metrics["mae_mwh"] == 1.5


def test_conformal_radius_uses_finite_sample_correction() -> None:
    assert conformal_radius(np.array([1.0, 2.0, 3.0, 4.0]), 0.8) == 4.0
    diagnostics = interval_diagnostics(
        actual=np.array([10.0, 12.0, 14.0]),
        prediction=np.array([11.0, 11.0, 15.0]),
        radii={"p80": 1.0, "p95": 2.0},
    )
    assert diagnostics.loc[diagnostics["interval"] == "p80", "observed_coverage"].iloc[0] == 1.0


def test_moving_block_bootstrap_is_deterministic_and_positive() -> None:
    selected = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    baseline = selected + 3.0
    first = moving_block_bootstrap_improvement(selected, baseline, iterations=50, block_length=3)
    second = moving_block_bootstrap_improvement(selected, baseline, iterations=50, block_length=3)
    assert first == second
    assert first["estimate_mwh"] == 3.0
