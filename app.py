from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import REPORTS_DIR, SQL_REPORTS_DIR

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
MODEL_LABELS = {
    "seasonal_naive_7d": "Seasonal naive (7d)",
    "seasonal_naive_364d": "Seasonal naive (364d)",
    "ridge_autoregressive": "Ridge autoregressive",
    "hgb_autoregressive": "HGB autoregressive",
    "hgb_weather_informed": "HGB + weather",
}

st.set_page_config(
    page_title="Spain Demand Forecast Lab",
    page_icon=":material/electric_bolt:",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1500px;}
    h1, h2, h3 {color: #17324D; letter-spacing: 0;}
    [data-testid="stMetricValue"] {color: #17324D;}
    [data-testid="stMetricLabel"] {font-size: 0.86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_csv(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / name, parse_dates=parse_dates)


def read_sql_report(name: str) -> pd.DataFrame:
    return pd.read_csv(SQL_REPORTS_DIR / f"{name}.csv")


def read_summary() -> dict[str, object]:
    return json.loads((REPORTS_DIR / "forecast_summary.json").read_text(encoding="utf-8"))


def forecast_figure(frame: pd.DataFrame, prediction_column: str, show_intervals: bool) -> go.Figure:
    figure = go.Figure()
    if show_intervals:
        figure.add_trace(
            go.Scatter(
                x=pd.concat([frame["date"], frame["date"].iloc[::-1]]),
                y=pd.concat([frame["upper_p95_mwh"], frame["lower_p95_mwh"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(37,99,235,0.10)",
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                name="95% interval",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=pd.concat([frame["date"], frame["date"].iloc[::-1]]),
                y=pd.concat([frame["upper_p80_mwh"], frame["lower_p80_mwh"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(37,99,235,0.18)",
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                name="80% interval",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["actual_demand_mwh"],
            mode="lines",
            line={"color": NAVY, "width": 2},
            name="Actual demand",
        )
    )
    model_name = prediction_column.removesuffix("_mwh")
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame[prediction_column],
            mode="lines",
            line={"color": MODEL_COLORS.get(model_name, AMBER), "width": 1.8},
            name=MODEL_LABELS.get(model_name, model_name.replace("_", " ")),
        )
    )
    figure.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=45, b=20),
        title="Temporal test forecast",
        xaxis_title="",
        yaxis_title="Daily demand (MWh)",
        legend={"orientation": "h", "y": 1.04, "x": 0},
        hovermode="x unified",
    )
    return figure


def show_overview(summary: dict[str, object]) -> None:
    predictions = read_csv("test_forecast_predictions.csv", parse_dates=["date"])
    registry = read_csv("model_registry.csv")
    intervals = read_csv("interval_diagnostics.csv")
    selected_model = str(summary["model_selection"]["selected_model"])
    selected = registry.loc[registry["model"] == selected_model].iloc[0]
    baseline = registry.loc[registry["model"] == "seasonal_naive_7d"].iloc[0]
    improvement = 1 - selected["mae_mwh"] / baseline["mae_mwh"]
    coverage_95 = intervals.loc[intervals["interval"] == "p95", "observed_coverage"].iloc[0]

    first, second, third, fourth = st.columns(4)
    first.metric("Selected model", MODEL_LABELS[selected_model])
    second.metric("Test MAE", f"{selected['mae_mwh']:,.0f} MWh")
    third.metric("MAE vs 7-day baseline", f"{improvement:.1%}")
    fourth.metric("95% interval coverage", f"{coverage_95:.1%}")

    control_left, control_right = st.columns([1.25, 1])
    model_names = registry["model"].tolist()
    with control_left:
        selected_display_model = st.selectbox(
            "Forecast series",
            model_names,
            index=model_names.index(selected_model),
            format_func=lambda model_name: MODEL_LABELS[model_name],
        )
    with control_right:
        minimum = predictions["date"].min().date()
        maximum = predictions["date"].max().date()
        date_range = st.date_input(
            "Test period",
            value=(predictions["date"].iloc[-180].date(), maximum),
            min_value=minimum,
            max_value=maximum,
        )
    if len(date_range) != 2:
        start_date, end_date = minimum, maximum
    else:
        start_date, end_date = date_range
    filtered = predictions.loc[predictions["date"].dt.date.between(start_date, end_date)].copy()
    st.plotly_chart(
        forecast_figure(
            filtered,
            f"{selected_display_model}_mwh",
            selected_display_model == selected_model,
        ),
        use_container_width=True,
    )

    residuals = predictions.assign(
        absolute_error_mwh=lambda frame: frame["selected_error_mwh"].abs(),
        error_direction=lambda frame: (
            frame["selected_error_mwh"].ge(0).map({True: "Over-forecast", False: "Under-forecast"})
        ),
    )
    residual_plot = px.scatter(
        residuals,
        x="temperature_mean_c",
        y="selected_error_mwh",
        color="error_direction",
        color_discrete_map={"Over-forecast": AMBER, "Under-forecast": BLUE},
        hover_data=["date", "actual_demand_mwh", "absolute_error_mwh", "is_holiday"],
        opacity=0.72,
    )
    residual_plot.add_hline(y=0, line_color=GREY, line_dash="dash")
    residual_plot.update_layout(
        title="Residuals against realized temperature",
        xaxis_title="Equal-weight city temperature proxy (C)",
        yaxis_title="Forecast error (MWh)",
        height=430,
        legend_title_text="",
    )
    st.plotly_chart(residual_plot, use_container_width=True)
    st.info(str(summary["weather_assumption"]))


def show_backtests(summary: dict[str, object]) -> None:
    folds = read_csv("backtest_folds.csv", parse_dates=["origin"])
    registry = read_csv("model_registry.csv")
    selected_model = str(summary["model_selection"]["selected_model"])
    figure = px.line(
        folds,
        x="origin",
        y="mae_mwh",
        color="model",
        markers=True,
        color_discrete_map=MODEL_COLORS,
        hover_data=["rmse_mwh", "smape_pct", "bias_mwh", "mase_7d"],
    )
    figure.update_layout(
        title="Expanding-window backtest",
        xaxis_title="2024 forecast origin",
        yaxis_title="28-day MAE (MWh)",
        legend_title_text="",
        height=500,
    )
    st.plotly_chart(figure, use_container_width=True)

    display = registry.copy()
    display["status"] = display["model"].eq(selected_model).map({True: "selected", False: "candidate"})
    st.dataframe(
        display[
            [
                "status",
                "model",
                "feature_count",
                "backtest_mae_mean_mwh",
                "backtest_mae_std_mwh",
                "mae_mwh",
                "rmse_mwh",
                "smape_pct",
                "bias_mwh",
                "uses_realized_weather",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


def show_uncertainty() -> None:
    intervals = read_csv("interval_diagnostics.csv")
    diagnostics = read_csv("residual_diagnostics.csv")
    comparison = read_csv("statistical_comparison.csv")

    coverage = intervals.melt(
        id_vars=["interval", "radius_mwh", "mean_width_mwh"],
        value_vars=["target_coverage", "observed_coverage"],
        var_name="coverage_type",
        value_name="coverage",
    )
    coverage_chart = px.bar(
        coverage,
        x="interval",
        y="coverage",
        color="coverage_type",
        barmode="group",
        text_auto=".1%",
        color_discrete_map={"target_coverage": GREY, "observed_coverage": TEAL},
    )
    coverage_chart.update_layout(
        title="Conformal interval calibration on 2025",
        yaxis_range=[0, 1.08],
        yaxis_tickformat=".0%",
        legend_title_text="",
        height=430,
    )
    left, right = st.columns([1.15, 1])
    left.plotly_chart(coverage_chart, use_container_width=True)
    right.dataframe(intervals, hide_index=True, use_container_width=True)

    monthly = diagnostics.loc[diagnostics["dimension"] == "month"].copy()
    monthly_chart = px.bar(
        monthly,
        x="cohort",
        y="mae_mwh",
        color="bias_mwh",
        color_continuous_scale=[[0, BLUE], [0.5, "#F8FAFC"], [1, RED]],
        color_continuous_midpoint=0,
        hover_data=["days", "rmse_mwh", "smape_pct", "wape_pct", "bias_mwh"],
    )
    monthly_chart.update_layout(
        title="Monthly test error | color encodes signed bias",
        xaxis_title="Month",
        yaxis_title="MAE (MWh)",
        height=450,
    )
    st.plotly_chart(monthly_chart, use_container_width=True)
    st.dataframe(comparison, hide_index=True, use_container_width=True)


def show_data() -> None:
    monthly_weather = read_sql_report("monthly_weather")
    sensitivity = read_sql_report("degree_day_sensitivity")
    calendar_profile = read_sql_report("calendar_demand_profile")
    monthly_weather["month"] = pd.to_datetime(monthly_weather["month"])

    weather_chart = go.Figure()
    weather_chart.add_trace(
        go.Scatter(
            x=monthly_weather["month"],
            y=monthly_weather["avg_temperature_c"],
            mode="lines",
            line={"color": RED, "width": 2},
            name="Temperature",
        )
    )
    weather_chart.add_trace(
        go.Bar(
            x=monthly_weather["month"],
            y=monthly_weather["precipitation_mm"],
            marker_color=BLUE,
            opacity=0.45,
            name="Precipitation",
            yaxis="y2",
        )
    )
    weather_chart.update_layout(
        title="Monthly climate proxy",
        yaxis={"title": "Mean temperature (C)"},
        yaxis2={"title": "Precipitation (mm)", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": 1.05},
        height=460,
    )
    st.plotly_chart(weather_chart, use_container_width=True)

    first, second = st.columns([1.4, 1])
    with first:
        sensitivity_chart = px.line(
            sensitivity,
            x="temperature_bin_c",
            y="avg_demand_mwh",
            markers=True,
            color_discrete_sequence=[TEAL],
        )
        sensitivity_chart.update_layout(
            title="Demand by 2 C temperature bin",
            xaxis_title="Temperature bin (C)",
            yaxis_title="Average demand (MWh)",
            height=390,
        )
        st.plotly_chart(sensitivity_chart, use_container_width=True)
    with second:
        st.dataframe(calendar_profile, hide_index=True, use_container_width=True)


st.title("Spain Demand Forecast Lab")
st.caption("Probabilistic forecasting, temporal validation and data-quality observability")

required_report = REPORTS_DIR / "forecast_summary.json"
if not required_report.exists():
    st.error("Run `python -m src.run_pipeline` before starting the dashboard.")
    st.stop()

run_summary = read_summary()
st.caption(
    f"Test: {run_summary['test_period']['start']} to {run_summary['test_period']['end']} | "
    f"Selected: {run_summary['model_selection']['selected_model']}"
)

overview_tab, backtest_tab, uncertainty_tab, data_tab = st.tabs(
    ["Overview", "Backtests", "Intervals", "Data"]
)
with overview_tab:
    show_overview(run_summary)
with backtest_tab:
    show_backtests(run_summary)
with uncertainty_tab:
    show_uncertainty()
with data_tab:
    show_data()
