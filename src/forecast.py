from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    BACKTEST_HORIZON_DAYS,
    BACKTEST_ORIGINS,
    FIGURES_DIR,
    FORECAST_FEATURES,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_START_DATE,
)


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


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_mwh": float(mean_absolute_error(actual, predicted)),
        "rmse_mwh": float(np.sqrt(mean_squared_error(actual, predicted))),
        "smape_pct": smape(actual, predicted),
    }


def make_hgb_model() -> HistGradientBoostingRegressor:
    """Create the non-linear forecast candidate with fixed, documented parameters."""
    return HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=260,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )


def evaluate_fold(train: pd.DataFrame, test: pd.DataFrame) -> list[dict[str, float | str]]:
    """Compare weekly seasonal naive and HGB on one chronological fold."""
    hgb = make_hgb_model()
    hgb.fit(train[FORECAST_FEATURES], train["demand_mwh"])
    hgb_prediction = hgb.predict(test[FORECAST_FEATURES])
    seasonal_naive_prediction = test["lag_7"].to_numpy()
    return [
        {"model": "seasonal_naive_7d", **regression_metrics(test["demand_mwh"], seasonal_naive_prediction)},
        {"model": "hist_gradient_boosting", **regression_metrics(test["demand_mwh"], hgb_prediction)},
    ]


def backtest(model_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fixed-origin 28-day backtests entirely before the final test period."""
    rows: list[dict[str, float | str]] = []
    features = model_features.copy()
    features["date"] = pd.to_datetime(features["date"])
    for origin_text in BACKTEST_ORIGINS:
        origin = pd.Timestamp(origin_text)
        test_end = origin + pd.Timedelta(days=BACKTEST_HORIZON_DAYS)
        train = features[features["date"] < origin]
        test = features[(features["date"] >= origin) & (features["date"] < test_end)]
        if len(train) == 0 or len(test) != BACKTEST_HORIZON_DAYS:
            raise ValueError(f"Invalid backtest fold at {origin.date()}")
        for result in evaluate_fold(train, test):
            rows.append({"origin": origin.date().isoformat(), **result})
    fold_results = pd.DataFrame(rows)
    summary = (
        fold_results.groupby("model", as_index=False)[["mae_mwh", "rmse_mwh", "smape_pct"]]
        .mean()
        .sort_values("mae_mwh")
        .reset_index(drop=True)
    )
    return fold_results, summary


def plot_backtest_mae(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    colors = ["#0F766E", "#F59E0B"]
    axis.bar(summary["model"], summary["mae_mwh"], color=colors[: len(summary)])
    axis.set_ylabel("MAE medio (MWh)")
    axis.set_title("Backtesting: error medio por modelo")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_test_forecast(predictions: pd.DataFrame, output_path: Path) -> None:
    display = predictions.tail(120)
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(display["date"], display["actual_demand_mwh"], label="Demanda real", color="#1D4ED8")
    axis.plot(
        display["date"],
        display["selected_prediction_mwh"],
        label="Prediccion seleccionada",
        color="#D97706",
    )
    axis.set_title("Forecast diario: ultimos 120 dias del test temporal")
    axis.set_ylabel("MWh")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_forecast(model_features: pd.DataFrame) -> dict[str, object]:
    """Select on backtests, then evaluate once on the 2025 temporal holdout."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    features = model_features.copy()
    features["date"] = pd.to_datetime(features["date"])
    test_start = pd.Timestamp(TEST_START_DATE)
    training = features[features["date"] < test_start]
    test = features[features["date"] >= test_start]
    if training.empty or test.empty:
        raise ValueError("Temporal train/test split is empty")

    fold_results, backtest_summary = backtest(training)
    fold_results.to_csv(REPORTS_DIR / "backtest_folds.csv", index=False)
    backtest_summary.to_csv(REPORTS_DIR / "backtest_summary.csv", index=False)
    selected_model = str(backtest_summary.iloc[0]["model"])

    hgb = make_hgb_model()
    hgb.fit(training[FORECAST_FEATURES], training["demand_mwh"])
    hgb_prediction = hgb.predict(test[FORECAST_FEATURES])
    seasonal_naive_prediction = test["lag_7"].to_numpy()
    test_metrics = pd.DataFrame(
        [
            {
                "model": "seasonal_naive_7d",
                **regression_metrics(test["demand_mwh"], seasonal_naive_prediction),
            },
            {
                "model": "hist_gradient_boosting",
                **regression_metrics(test["demand_mwh"], hgb_prediction),
            },
        ]
    ).sort_values("mae_mwh")
    test_metrics.to_csv(REPORTS_DIR / "test_metrics.csv", index=False)

    selected_prediction = (
        hgb_prediction if selected_model == "hist_gradient_boosting" else seasonal_naive_prediction
    )
    predictions = pd.DataFrame(
        {
            "date": test["date"].dt.date,
            "actual_demand_mwh": test["demand_mwh"].to_numpy(),
            "seasonal_naive_7d_mwh": seasonal_naive_prediction,
            "hist_gradient_boosting_mwh": hgb_prediction,
            "selected_prediction_mwh": selected_prediction,
        }
    )
    predictions["selected_error_mwh"] = (
        predictions["selected_prediction_mwh"] - predictions["actual_demand_mwh"]
    )
    predictions.to_csv(REPORTS_DIR / "test_forecast_predictions.csv", index=False)
    plot_backtest_mae(backtest_summary, FIGURES_DIR / "backtest_mae.png")
    plot_test_forecast(predictions, FIGURES_DIR / "test_forecast.png")

    if selected_model == "hist_gradient_boosting":
        joblib.dump(
            {
                "model": hgb,
                "feature_columns": FORECAST_FEATURES,
                "training_end": training["date"].max().date().isoformat(),
                "forecast_mode": "one-step-ahead with observed historical lags",
            },
            MODEL_PATH,
        )

    selected_test_metrics = test_metrics.loc[test_metrics["model"] == selected_model].iloc[0].to_dict()
    summary = {
        "forecast_mode": "One-step-ahead daily forecast using only lags ending before the target date.",
        "test_period": {
            "start": test["date"].min().date().isoformat(),
            "end": test["date"].max().date().isoformat(),
            "rows": int(len(test)),
        },
        "model_selection": {
            "criterion": "Lowest mean MAE across 2024 rolling backtests",
            "selected_model": selected_model,
        },
        "backtest_summary": backtest_summary.to_dict(orient="records"),
        "selected_test_metrics": {
            key: float(value) if isinstance(value, np.floating) else value
            for key, value in selected_test_metrics.items()
        },
    }
    (REPORTS_DIR / "forecast_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
