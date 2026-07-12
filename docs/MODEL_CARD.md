# Model Card

## Resumen

El artefacto `models/demand_hgb.joblib` contiene el modelo seleccionado `hgb_weather_informed`, una corrección de sesgo de 1.549 MWh y radios conformales para niveles 80% y 95%. Se entrenó sobre observaciones anteriores a 2025 y se evaluó una vez sobre todo 2025.

## Uso previsto

- Demostrar un flujo reproducible de forecasting diario con datos públicos.
- Comparar candidatos, intervalos y políticas de evaluación temporal.
- Explorar hipótesis de calendario y clima en un entorno académico o de portfolio.

## Uso no previsto

- Operación de red, compra/venta de energía, decisiones regulatorias o compromisos financieros.
- Forecast recursivo de semanas o meses sin actualizar los lags.
- Afirmar que el clima realizado ERA5 estaría disponible de igual forma en producción.

## Entradas

32 variables: calendario, festivos nacionales, estacionalidad Fourier, cinco lags de demanda, ventanas móviles y diez variables climáticas. Los lags están desplazados un día. El clima del objetivo es reanálisis realizado, una limitación central.

## Selección

Doce ventanas expansivas de 28 días durante 2024 comparan cinco candidatos. El modelo clima+HGB gana con MAE medio 9.997 MWh. No se emplea 2025 en selección, calibración de hiperparámetros ni corrección de sesgo.

## Rendimiento de test

| Métrica | HGB + weather |
|---|---:|
| MAE | 12.546 MWh |
| RMSE | 19.245 MWh |
| sMAPE | 1,82% |
| WAPE | 1,79% |
| MASE vs naive 7d | 0,382 |
| Sesgo | -2.644 MWh |

El baseline semanal obtiene MAE 32.816 MWh. El contraste Diebold-Mariano frente a ese baseline da `p = 2,9e-19`; el bootstrap de bloques semanales sitúa la mejora de MAE entre 16.044 y 24.506 MWh al 95%.

## Incertidumbre

Los radios se estiman a partir de 336 residuos fuera de muestra de 2024, después de una corrección de sesgo basada en la mediana. Cobertura observada en 2025:

| Intervalo | Objetivo | Observado |
|---|---:|---:|
| 80% | 80,0% | 72,1% |
| 95% | 95,0% | 94,0% |

La cobertura 80% por debajo del objetivo indica que la distribución de residuos cambió o que la hipótesis de estacionariedad es limitada. No se calibra de nuevo sobre test.

## Riesgos y mitigaciones

| Riesgo | Mitigación incluida |
|---|---|
| Fuga del objetivo | Lags y ventanas terminan en `t-1`. |
| Selección sobre test | Selección y corrección se hacen en 2024. |
| Clima no disponible en producción | Supuesto visible, modelo autoregresivo de comparación y limitación documentada. |
| Variabilidad serial | Bootstrap por bloques y Newey-West. |
| Revisiones de fuente | Snapshots, manifests y hashes. |
| Compatibilidad del artefacto | Dependencias fijadas y runtime registrado. |

## Limitaciones

No existe validación externa ni con forecasts meteorológicos archivados. Los festivos son nacionales, y el modelo no incorpora precios, generación, economía, restricciones o eventos de red. Las asociaciones predictivas no demuestran causalidad.
