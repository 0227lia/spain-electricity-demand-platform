from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#17324D"
BLUE = "#2563EB"
TEAL = "#0F766E"
AMBER = "#D97706"
RED = "#DC2626"
GREY = "#94A3B8"
MODEL_COLORS = {
    "seasonal_naive_7d": GREY,
    "seasonal_naive_364d": "#64748B",
    "ridge_autoregressive": AMBER,
    "hgb_autoregressive": BLUE,
    "hgb_weather_informed": TEAL,
}


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9E2EC", linewidth=0.8, alpha=0.8)
    axis.tick_params(colors="#334E68")
    axis.title.set_color(NAVY)


def plot_forecast_control_center(
    predictions: pd.DataFrame,
    backtest_folds: pd.DataFrame,
    intervals: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render a five-panel forecasting and uncertainty control center."""
    forecast = predictions.copy()
    forecast["date"] = pd.to_datetime(forecast["date"])
    recent = forecast.tail(180)
    figure = plt.figure(figsize=(18, 13), facecolor="white")
    grid = figure.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], hspace=0.42, wspace=0.36)
    forecast_axis = figure.add_subplot(grid[0, :])
    backtest_axis = figure.add_subplot(grid[1, 0])
    coverage_axis = figure.add_subplot(grid[1, 1])
    cohort_axis = figure.add_subplot(grid[2, 0])
    importance_axis = figure.add_subplot(grid[2, 1])

    forecast_axis.fill_between(
        recent["date"],
        recent["lower_p95_mwh"],
        recent["upper_p95_mwh"],
        color=BLUE,
        alpha=0.10,
        label="95% conformal interval",
    )
    forecast_axis.fill_between(
        recent["date"],
        recent["lower_p80_mwh"],
        recent["upper_p80_mwh"],
        color=BLUE,
        alpha=0.20,
        label="80% conformal interval",
    )
    forecast_axis.plot(
        recent["date"],
        recent["actual_demand_mwh"],
        color=NAVY,
        linewidth=1.7,
        label="Actual demand",
    )
    forecast_axis.plot(
        recent["date"],
        recent["selected_prediction_mwh"],
        color=AMBER,
        linewidth=1.5,
        label="Selected forecast",
    )
    forecast_axis.set_title(
        "Temporal test | Last 180 days with calibrated uncertainty", loc="left", weight="bold"
    )
    forecast_axis.set_ylabel("Daily demand (MWh)")
    forecast_axis.xaxis.set_major_locator(mdates.MonthLocator())
    forecast_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    forecast_axis.legend(ncol=4, frameon=False, loc="upper left")
    _style_axis(forecast_axis)

    for model_name, group in backtest_folds.groupby("model", sort=False):
        ordered = group.sort_values("origin")
        backtest_axis.plot(
            pd.to_datetime(ordered["origin"]),
            ordered["mae_mwh"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=MODEL_COLORS[model_name],
            label=model_name.replace("_", " "),
        )
    backtest_axis.set_title("12-fold expanding-window backtest", loc="left", weight="bold")
    backtest_axis.set_ylabel("MAE (MWh)")
    backtest_axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    backtest_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    backtest_axis.legend(frameon=False, fontsize=8, ncol=2)
    _style_axis(backtest_axis)

    positions = np.arange(len(intervals))
    width = 0.34
    coverage_axis.bar(
        positions - width / 2,
        intervals["target_coverage"],
        width,
        color=GREY,
        label="Target",
    )
    coverage_axis.bar(
        positions + width / 2,
        intervals["observed_coverage"],
        width,
        color=TEAL,
        label="Observed",
    )
    for index, value in enumerate(intervals["observed_coverage"]):
        coverage_axis.text(index + width / 2, value + 0.015, f"{value:.1%}", ha="center", fontsize=9)
    coverage_axis.set_xticks(positions, intervals["interval"].str.upper())
    coverage_axis.set_ylim(0, 1.08)
    coverage_axis.set_title("Prediction interval calibration", loc="left", weight="bold")
    coverage_axis.set_ylabel("Coverage")
    coverage_axis.legend(frameon=False)
    _style_axis(coverage_axis)

    monthly = residual_diagnostics.loc[residual_diagnostics["dimension"] == "month"].copy()
    monthly = monthly.sort_values("cohort")
    colors = [RED if value < 0 else TEAL for value in monthly["bias_mwh"]]
    cohort_axis.bar(monthly["cohort"], monthly["mae_mwh"], color=BLUE, alpha=0.88)
    cohort_axis.set_title("2025 monthly error and bias", loc="left", weight="bold")
    cohort_axis.set_xlabel("Month")
    cohort_axis.set_ylabel("MAE (MWh)", color=BLUE)
    bias_axis = cohort_axis.twinx()
    bias_axis.plot(monthly["cohort"], monthly["bias_mwh"], color=AMBER, marker="o", linewidth=1.8)
    bias_axis.axhline(0, color="#64748B", linewidth=0.9)
    bias_axis.set_ylabel("")
    bias_axis.tick_params(axis="y", colors=AMBER)
    for spine in ["top"]:
        bias_axis.spines[spine].set_visible(False)
    _style_axis(cohort_axis)
    for patch, color in zip(cohort_axis.patches, colors, strict=True):
        patch.set_edgecolor(color)
        patch.set_linewidth(0.8)

    top_importance = feature_importance.head(12).sort_values("importance_mean_mwh")
    if top_importance.empty:
        importance_axis.text(0.5, 0.5, "Selected baseline has no fitted feature model", ha="center")
    else:
        importance_axis.barh(
            top_importance["feature"],
            top_importance["importance_mean_mwh"],
            xerr=top_importance["importance_std_mwh"],
            color=TEAL,
            alpha=0.9,
            error_kw={"ecolor": "#334E68", "capsize": 2},
        )
    importance_axis.set_title("Post-hoc permutation importance on test", loc="left", weight="bold")
    importance_axis.set_xlabel("Increase in MAE when permuted (MWh)")
    _style_axis(importance_axis)

    figure.suptitle(
        "Spain Electricity Demand | Probabilistic Forecast Control Center",
        color=NAVY,
        fontsize=21,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.957,
        "Model selection on 2024 expanding windows | Final evaluation on untouched 2025",
        ha="center",
        color="#526777",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_forecast_scorecard(model_registry: pd.DataFrame, output_path: Path) -> None:
    """Render a compact model registry table for the README and portfolio."""
    display = model_registry[
        [
            "model",
            "feature_count",
            "backtest_mae_mean_mwh",
            "backtest_mae_std_mwh",
            "mae_mwh",
            "smape_pct",
            "selected",
        ]
    ].copy()
    display.columns = ["Model", "Features", "BT MAE", "BT SD", "Test MAE", "Test sMAPE", "Selected"]
    display["Model"] = display["Model"].str.replace("_", " ")
    for column in ["BT MAE", "BT SD", "Test MAE"]:
        display[column] = display[column].map(lambda value: f"{value:,.0f}")
    display["Test sMAPE"] = display["Test sMAPE"].map(lambda value: f"{value:.2f}%")
    display["Selected"] = display["Selected"].map({True: "Yes", False: ""})

    figure, axis = plt.subplots(figsize=(15, 3.9), facecolor="white")
    axis.axis("off")
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.25, 0.09, 0.13, 0.11, 0.13, 0.14, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.65)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        else:
            is_selected = display.iloc[row - 1]["Selected"] == "Yes"
            cell.set_facecolor("#DFF4EE" if is_selected else ("#F1F5F9" if row % 2 else "#E2E8F0"))
            cell.set_text_props(color=NAVY, weight="bold" if is_selected else "normal")
    axis.set_title(
        "Model Registry | Backtest selection and untouched temporal test",
        color=NAVY,
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
