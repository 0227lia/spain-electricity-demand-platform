# Diccionario de datos

## `daily_demand.csv`

| Campo | Descripción |
|---|---|
| `date` | Fecha local de la observación diaria. |
| `demand_mwh` | Demanda diaria nacional en MWh publicada por REE. |
| `year`, `month`, `week`, `day_of_week` | Atributos de calendario. |
| `is_weekend` | Sábado o domingo. |
| `is_holiday` | Festivo nacional según `python-holidays`. |
| `is_day_before_holiday`, `is_day_after_holiday` | Día laborable adyacente a festivo. |
| `is_bridge_day` | Laborable entre festivo y fin de semana. |

## `city_weather.csv`

Una fila por ciudad y día. Incluye `city`, `weight`, temperaturas a 2m, precipitación, viento máximo y radiación corta diaria de ERA5.

| Campo | Unidad |
|---|---|
| `temperature_mean_c`, `temperature_min_c`, `temperature_max_c` | °C |
| `precipitation_mm` | mm/día |
| `wind_speed_max_kmh` | km/h |
| `solar_radiation_mj_m2` | MJ/m²/día |

## `national_weather_proxy.csv`

Agrega las cinco ciudades con pesos iguales. Además de las variables anteriores, contiene:

| Campo | Definición |
|---|---|
| `temperature_range_c` | Temperatura máxima media menos mínima media. |
| `city_temperature_spread_c` | Máxima menos mínima entre temperaturas medias de las cinco ciudades. |
| `heating_degree_days` | `max(18 - temperature_mean_c, 0)`. |
| `cooling_degree_days` | `max(temperature_mean_c - 22, 0)`. |

## `model_features.csv`

Incluye demanda, calendario y proxy meteorológico, más:

| Grupo | Campos |
|---|---|
| Estacionalidad | `day_of_year_sin/cos`, `week_of_year_sin/cos` |
| Lags | `lag_1`, `lag_7`, `lag_14`, `lag_28`, `lag_364` |
| Ventanas | `rolling_mean_7/28/56`, `rolling_std_7/28` |
| Cambio corto | `lag_1_minus_7` |

Todos los lags y ventanas terminan en `t-1`. Las variables climáticas del día `t` corresponden al reanálisis realizado y se documentan como una hipótesis de disponibilidad meteorológica.

## Informes principales

| Archivo | Contenido |
|---|---|
| `backtest_folds.csv` | Métricas de las 12 x 5 evaluaciones temporales. |
| `backtest_predictions.csv` | Predicciones fuera de muestra de 2024 para calibración. |
| `model_registry.csv` | Registro de candidatos, features y métricas de test. |
| `test_forecast_predictions.csv` | Forecast de 2025, intervalos y errores. |
| `interval_diagnostics.csv` | Cobertura, radio y anchura de p80/p95. |
| `statistical_comparison.csv` | Diebold-Mariano y bootstrap de bloques. |
| `residual_diagnostics.csv` | Error por mes, día y régimen térmico. |
