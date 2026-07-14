from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

NAVY = "#0B2130"
BLUE = "#2F6BFF"
TEAL = "#0F766E"
AMBER = "#D28A1E"
RED = "#E85D45"
GREY = "#94A3B8"
MUTED = "#597181"
GRID = "#D6E0E5"
PAPER = "#F4F7F6"
MODEL_COLORS = {
    "seasonal_naive_7d": GREY,
    "seasonal_naive_364d": "#64748B",
    "ridge_autoregressive": AMBER,
    "hgb_autoregressive": BLUE,
    "hgb_weather_informed": TEAL,
}


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    axis.tick_params(colors=MUTED)
    axis.title.set_color(NAVY)
    axis.set_axisbelow(True)


def plot_forecast_control_center(
    predictions: pd.DataFrame,
    backtest_folds: pd.DataFrame,
    intervals: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_path: Path,
) -> None:
    """Render a five-panel forecasting and uncertainty control center."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": NAVY,
            "axes.edgecolor": GRID,
            "axes.facecolor": "#FFFFFF",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    forecast = predictions.copy()
    forecast["date"] = pd.to_datetime(forecast["date"])
    recent = forecast.tail(180)
    figure = plt.figure(figsize=(16, 9), dpi=170, facecolor=PAPER)
    grid = figure.add_gridspec(
        2, 6, left=0.065, right=0.97, top=0.61, bottom=0.1, height_ratios=[1.15, 1], hspace=0.48, wspace=0.72
    )
    forecast_axis = figure.add_subplot(grid[0, :4])
    coverage_axis = figure.add_subplot(grid[0, 4:])
    backtest_axis = figure.add_subplot(grid[1, :2])
    cohort_axis = figure.add_subplot(grid[1, 2:4])
    importance_axis = figure.add_subplot(grid[1, 4:])

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
        linewidth=1.8,
        label="Actual demand",
    )
    forecast_axis.plot(
        recent["date"],
        recent["selected_prediction_mwh"],
        color=AMBER,
        linewidth=1.6,
        label="Selected forecast",
    )
    forecast_axis.set_title("Untouched 2025 test  |  latest 180 days", loc="left", color=NAVY, pad=10)
    forecast_axis.set_ylabel("Daily demand (MWh)")
    forecast_axis.xaxis.set_major_locator(mdates.MonthLocator())
    forecast_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    forecast_axis.legend(ncol=2, frameon=False, loc="upper left", fontsize=7.8)
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
    backtest_axis.set_title("Expanding-window backtest", loc="left", color=NAVY, pad=10)
    backtest_axis.set_ylabel("MAE (MWh)")
    backtest_axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    backtest_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    backtest_axis.legend(frameon=False, fontsize=6.8, ncol=1, loc="upper right")
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
    coverage_axis.set_title("Interval calibration", loc="left", color=NAVY, pad=10)
    coverage_axis.set_ylabel("Coverage")
    coverage_axis.legend(frameon=False, fontsize=7.5)
    _style_axis(coverage_axis)

    monthly = residual_diagnostics.loc[residual_diagnostics["dimension"] == "month"].copy()
    monthly = monthly.sort_values("cohort")
    colors = [RED if value < 0 else TEAL for value in monthly["bias_mwh"]]
    cohort_axis.bar(monthly["cohort"], monthly["mae_mwh"], color=BLUE, alpha=0.88)
    cohort_axis.set_title("Monthly error and signed bias", loc="left", color=NAVY, pad=10)
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

    top_importance = feature_importance.head(8).sort_values("importance_mean_mwh")
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
    importance_axis.set_title("Permutation importance on test", loc="left", color=NAVY, pad=10)
    importance_axis.set_xlabel("Increase in MAE when permuted (MWh)")
    _style_axis(importance_axis)

    test_mae = float((forecast["actual_demand_mwh"] - forecast["selected_prediction_mwh"]).abs().mean())
    baseline_mae = float((forecast["actual_demand_mwh"] - forecast["seasonal_naive_7d_mwh"]).abs().mean())
    mae_reduction = 1 - test_mae / baseline_mae
    p95_row = intervals.loc[intervals["interval"].str.lower().eq("p95")]
    p95_coverage = float(p95_row["observed_coverage"].iloc[0]) if not p95_row.empty else float("nan")
    folds = int(backtest_folds["origin"].nunique())
    kpis = [
        ("TEST MAE", f"{test_mae / 1000:.1f}k MWh", "untouched 2025"),
        ("VS 7-DAY BASELINE", f"{mae_reduction:.1%}", "lower MAE"),
        ("P95 COVERAGE", f"{p95_coverage:.1%}", "observed on test"),
        ("BACKTEST WINDOWS", f"{folds}", "expanding 28-day folds"),
    ]
    card_width = 0.205
    for index, (label, value, note) in enumerate(kpis):
        left = 0.065 + index * 0.225
        figure.patches.append(
            Rectangle(
                (left, 0.665),
                card_width,
                0.115,
                transform=figure.transFigure,
                facecolor="white",
                edgecolor=GRID,
                linewidth=0.8,
            )
        )
        figure.text(left + 0.012, 0.748, label, color=TEAL, fontsize=8, weight="bold")
        figure.text(left + 0.012, 0.704, value, color=NAVY, fontsize=18, weight="bold")
        figure.text(left + 0.012, 0.68, note, color=MUTED, fontsize=7.5)

    figure.text(0.065, 0.945, "SPAIN ELECTRICITY DEMAND PLATFORM", color=TEAL, fontsize=10, weight="bold")
    figure.text(
        0.065,
        0.885,
        "Probabilistic forecasts with visible failure modes",
        color=NAVY,
        fontsize=24,
        weight="bold",
    )
    figure.text(
        0.065,
        0.842,
        (
            "Model selection on 2024 expanding windows, calibrated uncertainty "
            "and final evaluation on untouched 2025."
        ),
        color=MUTED,
        fontsize=11,
    )
    figure.text(
        0.065,
        0.04,
        (
            "One-step daily forecasts. ERA5 target-day weather is a reproducible proxy, "
            "not a historical weather-forecast archive; intervals remain sample-specific."
        ),
        color=MUTED,
        fontsize=8.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=PAPER)
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
