from __future__ import annotations

import json
import math
import platform
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    AUTOREGRESSIVE_FEATURES,
    BACKTEST_HORIZON_DAYS,
    BACKTEST_ORIGINS,
    FIGURES_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_START_DATE,
    WEATHER_INFORMED_FEATURES,
)
from src.visualization import plot_forecast_control_center, render_forecast_scorecard

MODEL_FEATURES = {
    "ridge_autoregressive": AUTOREGRESSIVE_FEATURES,
    "hgb_autoregressive": AUTOREGRESSIVE_FEATURES,
    "hgb_weather_informed": WEATHER_INFORMED_FEATURES,
}
MODEL_NAMES = [
    "seasonal_naive_7d",
    "seasonal_naive_364d",
    "ridge_autoregressive",
    "hgb_autoregressive",
    "hgb_weather_informed",
]


def smape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """Return symmetric mean absolute percentage error in percent."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual_array) + np.abs(predicted_array)
    terms = np.divide(
        2 * np.abs(actual_array - predicted_array),
        denominator,
        out=np.zeros_like(actual_array),
        where=denominator != 0,
    )
    return float(100 * np.mean(terms))


def regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    seasonal_scale: float | None = None,
) -> dict[str, float]:
    """Calculate scale, percentage, and bias metrics for one forecast vector."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    errors = predicted_array - actual_array
    mae = float(mean_absolute_error(actual_array, predicted_array))
    return {
        "mae_mwh": mae,
        "rmse_mwh": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "smape_pct": smape(actual_array, predicted_array),
        "wape_pct": float(100 * np.abs(errors).sum() / np.abs(actual_array).sum()),
        "bias_mwh": float(errors.mean()),
        "mase_7d": float(mae / seasonal_scale) if seasonal_scale and seasonal_scale > 0 else np.nan,
    }


def make_estimator(model_name: str) -> Any:
    """Create a documented candidate estimator."""
    if model_name == "ridge_autoregressive":
        return make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    if model_name in {"hgb_autoregressive", "hgb_weather_informed"}:
        return HistGradientBoostingRegressor(
            learning_rate=0.045,
            max_iter=320,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.3,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"No estimator is defined for {model_name}")


def fit_predict_candidate(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, Any | None]:
    """Fit one candidate or evaluate a deterministic seasonal baseline."""
    if model_name == "seasonal_naive_7d":
        return test["lag_7"].to_numpy(), None
    if model_name == "seasonal_naive_364d":
        return test["lag_364"].to_numpy(), None
    estimator = make_estimator(model_name)
    feature_columns = MODEL_FEATURES[model_name]
    estimator.fit(train[feature_columns], train["demand_mwh"])
    return estimator.predict(test[feature_columns]), estimator


def evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    origin: pd.Timestamp,
) -> tuple[list[dict[str, float | str]], pd.DataFrame]:
    """Evaluate every candidate on one chronological, expanding-window fold."""
    metrics: list[dict[str, float | str]] = []
    predictions: list[pd.DataFrame] = []
    seasonal_scale = float(mean_absolute_error(test["demand_mwh"], test["lag_7"]))
    for model_name in MODEL_NAMES:
        prediction, _ = fit_predict_candidate(model_name, train, test)
        metrics.append(
            {
                "origin": origin.date().isoformat(),
                "model": model_name,
                **regression_metrics(test["demand_mwh"], prediction, seasonal_scale),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "origin": origin.date().isoformat(),
                    "date": test["date"].dt.date,
                    "model": model_name,
                    "actual_demand_mwh": test["demand_mwh"].to_numpy(),
                    "prediction_mwh": prediction,
                    "error_mwh": prediction - test["demand_mwh"].to_numpy(),
                }
            )
        )
    return metrics, pd.concat(predictions, ignore_index=True)


def backtest(model_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run 12 fixed-origin 28-day backtests entirely before the final test period."""
    metric_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    features = model_features.copy()
    features["date"] = pd.to_datetime(features["date"])
    for origin_text in BACKTEST_ORIGINS:
        origin = pd.Timestamp(origin_text)
        test_end = origin + pd.Timedelta(days=BACKTEST_HORIZON_DAYS)
        train = features[features["date"] < origin]
        test = features[(features["date"] >= origin) & (features["date"] < test_end)]
        if len(train) == 0 or len(test) != BACKTEST_HORIZON_DAYS:
            raise ValueError(f"Invalid backtest fold at {origin.date()}")
        fold_metrics, fold_predictions = evaluate_fold(train, test, origin)
        metric_rows.extend(fold_metrics)
        prediction_frames.append(fold_predictions)

    fold_results = pd.DataFrame(metric_rows)
    summary = (
        fold_results.groupby("model", as_index=False)
        .agg(
            backtest_mae_mean_mwh=("mae_mwh", "mean"),
            backtest_mae_std_mwh=("mae_mwh", "std"),
            backtest_rmse_mean_mwh=("rmse_mwh", "mean"),
            backtest_smape_mean_pct=("smape_pct", "mean"),
            backtest_wape_mean_pct=("wape_pct", "mean"),
            backtest_bias_mean_mwh=("bias_mwh", "mean"),
            folds=("origin", "count"),
        )
        .sort_values("backtest_mae_mean_mwh")
        .reset_index(drop=True)
    )
    return fold_results, summary, pd.concat(prediction_frames, ignore_index=True)


def conformal_radius(absolute_errors: pd.Series | np.ndarray, coverage: float) -> float:
    """Return the finite-sample corrected symmetric conformal radius."""
    errors = np.asarray(absolute_errors, dtype=float)
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    if len(errors) < 2 or not np.isfinite(errors).all():
        raise ValueError("at least two finite calibration errors are required")
    quantile_level = min(1.0, np.ceil((len(errors) + 1) * coverage) / len(errors))
    return float(np.quantile(errors, quantile_level, method="higher"))


def interval_diagnostics(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
    radii: dict[str, float],
) -> pd.DataFrame:
    """Measure empirical coverage and width for symmetric prediction intervals."""
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    rows: list[dict[str, float | str]] = []
    for label, radius in radii.items():
        target = float(label.removeprefix("p")) / 100
        lower = prediction_array - radius
        upper = prediction_array + radius
        rows.append(
            {
                "interval": label,
                "target_coverage": target,
                "observed_coverage": float(np.mean((actual_array >= lower) & (actual_array <= upper))),
                "radius_mwh": float(radius),
                "mean_width_mwh": float(2 * radius),
            }
        )
    return pd.DataFrame(rows)


def diebold_mariano_absolute(
    selected_absolute_error: pd.Series | np.ndarray,
    baseline_absolute_error: pd.Series | np.ndarray,
    max_lag: int = 7,
) -> dict[str, float]:
    """Compare absolute losses with a Newey-West long-run variance estimate."""
    selected = np.asarray(selected_absolute_error, dtype=float)
    baseline = np.asarray(baseline_absolute_error, dtype=float)
    if len(selected) != len(baseline) or len(selected) <= max_lag:
        raise ValueError("Loss vectors must have equal length greater than max_lag")
    loss_advantage = baseline - selected
    centered = loss_advantage - loss_advantage.mean()
    count = len(centered)
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, max_lag + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2 * (1 - lag / (max_lag + 1)) * covariance
    standard_error = np.sqrt(max(long_run_variance, 0) / count)
    statistic = float(loss_advantage.mean() / standard_error) if standard_error > 0 else np.inf
    p_value = float(math.erfc(abs(statistic) / np.sqrt(2)))
    return {
        "mean_absolute_error_advantage_mwh": float(loss_advantage.mean()),
        "dm_statistic": statistic,
        "two_sided_p_value": p_value,
        "newey_west_max_lag": float(max_lag),
    }


def moving_block_bootstrap_improvement(
    selected_absolute_error: pd.Series | np.ndarray,
    baseline_absolute_error: pd.Series | np.ndarray,
    iterations: int = 1000,
    block_length: int = 7,
) -> dict[str, float]:
    """Estimate a confidence interval for weekly-block MAE improvement."""
    selected = np.asarray(selected_absolute_error, dtype=float)
    baseline = np.asarray(baseline_absolute_error, dtype=float)
    if len(selected) != len(baseline) or len(selected) < block_length:
        raise ValueError("Error vectors must have equal length and at least one full block")
    advantages = baseline - selected
    rng = np.random.default_rng(RANDOM_STATE)
    starts = np.arange(0, len(advantages) - block_length + 1)
    estimates = np.empty(iterations)
    blocks_needed = int(np.ceil(len(advantages) / block_length))
    for index in range(iterations):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([advantages[start : start + block_length] for start in sampled_starts])[
            : len(advantages)
        ]
        estimates[index] = sample.mean()
    return {
        "estimate_mwh": float(advantages.mean()),
        "ci_lower_95_mwh": float(np.quantile(estimates, 0.025)),
        "ci_upper_95_mwh": float(np.quantile(estimates, 0.975)),
        "iterations": float(iterations),
        "block_length_days": float(block_length),
    }


def build_residual_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-model error by calendar and temperature regimes."""
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month.astype(str).str.zfill(2)
    frame["weekday"] = frame["date"].dt.day_name()
    frame["temperature_regime"] = pd.cut(
        frame["temperature_mean_c"],
        bins=[-np.inf, 8, 16, 24, np.inf],
        labels=["cold_below_8c", "cool_8_to_16c", "mild_16_to_24c", "hot_24c_plus"],
        right=False,
    ).astype(str)
    rows: list[dict[str, float | int | str]] = []
    for dimension in ["month", "weekday", "temperature_regime"]:
        for cohort, group in frame.groupby(dimension, observed=True):
            metrics = regression_metrics(group["actual_demand_mwh"], group["selected_prediction_mwh"])
            rows.append(
                {
                    "dimension": dimension,
                    "cohort": str(cohort),
                    "days": int(len(group)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _feature_importance(
    estimator: Any | None,
    selected_model: str,
    test: pd.DataFrame,
) -> pd.DataFrame:
    if estimator is None:
        return pd.DataFrame(columns=["feature", "importance_mean_mwh", "importance_std_mwh"])
    features = MODEL_FEATURES[selected_model]
    result = permutation_importance(
        estimator,
        test[features],
        test["demand_mwh"],
        scoring="neg_mean_absolute_error",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features,
                "importance_mean_mwh": result.importances_mean,
                "importance_std_mwh": result.importances_std,
            }
        )
        .sort_values("importance_mean_mwh", ascending=False)
        .reset_index(drop=True)
    )


def run_forecast(model_features: pd.DataFrame) -> dict[str, object]:
    """Select on pre-test folds, calibrate intervals, then evaluate once on 2025."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    features = model_features.copy()
    features["date"] = pd.to_datetime(features["date"])
    missing_features = sorted(set(WEATHER_INFORMED_FEATURES) - set(features))
    if missing_features:
        raise ValueError(f"Forecast features are missing: {missing_features}")
    test_start = pd.Timestamp(TEST_START_DATE)
    training = features[features["date"] < test_start]
    test = features[features["date"] >= test_start]
    if training.empty or test.empty:
        raise ValueError("Temporal train/test split is empty")

    fold_results, backtest_summary, backtest_predictions = backtest(training)
    selected_model = str(backtest_summary.iloc[0]["model"])
    selected_oof = backtest_predictions.loc[backtest_predictions["model"] == selected_model].copy()
    bias_correction_mwh = float(-selected_oof["error_mwh"].median())
    calibrated_oof_errors = selected_oof["error_mwh"] + bias_correction_mwh
    radii = {
        "p80": conformal_radius(calibrated_oof_errors.abs(), 0.80),
        "p95": conformal_radius(calibrated_oof_errors.abs(), 0.95),
    }

    test_prediction_columns: dict[str, np.ndarray] = {}
    fitted_estimators: dict[str, Any] = {}
    test_metric_rows: list[dict[str, float | str]] = []
    seasonal_scale = float(mean_absolute_error(test["demand_mwh"], test["lag_7"]))
    for model_name in MODEL_NAMES:
        prediction, estimator = fit_predict_candidate(model_name, training, test)
        test_prediction_columns[model_name] = prediction
        if estimator is not None:
            fitted_estimators[model_name] = estimator
        test_metric_rows.append(
            {
                "model": model_name,
                **regression_metrics(test["demand_mwh"], prediction, seasonal_scale),
            }
        )
    test_metrics = pd.DataFrame(test_metric_rows).sort_values("mae_mwh").reset_index(drop=True)
    selected_prediction = test_prediction_columns[selected_model] + bias_correction_mwh
    test_prediction_columns[selected_model] = selected_prediction
    corrected_metrics = regression_metrics(test["demand_mwh"], selected_prediction, seasonal_scale)
    for metric, value in corrected_metrics.items():
        test_metrics.loc[test_metrics["model"] == selected_model, metric] = value
    test_metrics = test_metrics.sort_values("mae_mwh").reset_index(drop=True)

    predictions = pd.DataFrame(
        {
            "date": test["date"].dt.date,
            "actual_demand_mwh": test["demand_mwh"].to_numpy(),
            "temperature_mean_c": test["temperature_mean_c"].to_numpy(),
            "is_holiday": test["is_holiday"].to_numpy(),
        }
    )
    for model_name, prediction in test_prediction_columns.items():
        predictions[f"{model_name}_mwh"] = prediction
    predictions["selected_model"] = selected_model
    predictions["selected_prediction_mwh"] = selected_prediction
    predictions["selected_error_mwh"] = selected_prediction - predictions["actual_demand_mwh"]
    for label, radius in radii.items():
        predictions[f"lower_{label}_mwh"] = selected_prediction - radius
        predictions[f"upper_{label}_mwh"] = selected_prediction + radius

    intervals = interval_diagnostics(test["demand_mwh"], selected_prediction, radii)
    residual_diagnostics = build_residual_diagnostics(predictions)
    selected_estimator = fitted_estimators.get(selected_model)
    feature_importance = _feature_importance(selected_estimator, selected_model, test)

    selected_absolute_error = np.abs(selected_prediction - test["demand_mwh"].to_numpy())
    baseline_absolute_error = np.abs(
        test_prediction_columns["seasonal_naive_7d"] - test["demand_mwh"].to_numpy()
    )
    dm_result = diebold_mariano_absolute(selected_absolute_error, baseline_absolute_error)
    bootstrap_result = moving_block_bootstrap_improvement(
        selected_absolute_error,
        baseline_absolute_error,
    )
    statistical_comparison = pd.DataFrame(
        [
            {"method": "Diebold-Mariano absolute loss", **dm_result},
            {"method": "Moving-block bootstrap", **bootstrap_result},
        ]
    )

    model_registry = backtest_summary.merge(test_metrics, on="model", how="left")
    model_registry["feature_count"] = (
        model_registry["model"]
        .map({name: len(columns) for name, columns in MODEL_FEATURES.items()})
        .fillna(1)
        .astype(int)
    )
    model_registry["selected"] = model_registry["model"].eq(selected_model)
    model_registry["uses_realized_weather"] = model_registry["model"].eq("hgb_weather_informed")

    fold_results.to_csv(REPORTS_DIR / "backtest_folds.csv", index=False)
    backtest_summary.to_csv(REPORTS_DIR / "backtest_summary.csv", index=False)
    backtest_predictions.to_csv(REPORTS_DIR / "backtest_predictions.csv", index=False)
    test_metrics.to_csv(REPORTS_DIR / "test_metrics.csv", index=False)
    predictions.to_csv(REPORTS_DIR / "test_forecast_predictions.csv", index=False)
    intervals.to_csv(REPORTS_DIR / "interval_diagnostics.csv", index=False)
    residual_diagnostics.to_csv(REPORTS_DIR / "residual_diagnostics.csv", index=False)
    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    model_registry.to_csv(REPORTS_DIR / "model_registry.csv", index=False)
    statistical_comparison.to_csv(REPORTS_DIR / "statistical_comparison.csv", index=False)

    selected_metrics = test_metrics.loc[test_metrics["model"] == selected_model].iloc[0].to_dict()
    runtime = {
        "python": platform.python_version(),
        "scikit_learn": sklearn_version,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    bundle = {
        "model": selected_estimator,
        "selected_model": selected_model,
        "feature_columns": MODEL_FEATURES.get(selected_model, []),
        "training_end": training["date"].max().date().isoformat(),
        "conformal_radii_mwh": radii,
        "bias_correction_mwh": bias_correction_mwh,
        "forecast_mode": "one-step-ahead with observed demand lags",
        "weather_assumption": (
            "The weather-informed candidate uses realized ERA5 weather as a proxy for a "
            "target-day weather forecast."
        ),
        "runtime": runtime,
        "test_metrics": selected_metrics,
    }
    joblib.dump(bundle, MODEL_PATH)

    plot_forecast_control_center(
        predictions,
        fold_results,
        intervals,
        residual_diagnostics,
        feature_importance,
        FIGURES_DIR / "forecast_control_center.png",
    )
    render_forecast_scorecard(model_registry, FIGURES_DIR / "model_scorecard.png")

    summary = {
        "forecast_mode": "One-step-ahead daily forecast using lags ending before the target date.",
        "weather_assumption": bundle["weather_assumption"],
        "runtime": runtime,
        "test_period": {
            "start": test["date"].min().date().isoformat(),
            "end": test["date"].max().date().isoformat(),
            "rows": int(len(test)),
        },
        "model_selection": {
            "criterion": "Lowest mean MAE across 12 expanding-window backtests in 2024",
            "selected_model": selected_model,
            "calibration_residuals": int(len(selected_oof)),
            "bias_correction_mwh": bias_correction_mwh,
        },
        "selected_test_metrics": {
            key: float(value) if isinstance(value, np.floating) else value
            for key, value in selected_metrics.items()
        },
        "prediction_intervals": intervals.to_dict(orient="records"),
        "statistical_comparison_vs_seasonal_naive_7d": {
            "diebold_mariano": dm_result,
            "moving_block_bootstrap": bootstrap_result,
        },
        "limitations": [
            "Realized ERA5 weather is not a historical target-day weather forecast and can "
            "overstate operational value.",
            "Symmetric conformal intervals assume 2024 out-of-fold residuals are informative for 2025.",
            "The task is one-step-ahead with observed lags, not a recursive long-horizon forecast.",
        ],
    }
    (REPORTS_DIR / "forecast_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
