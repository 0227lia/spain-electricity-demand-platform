# Metodología

## 1. Objetivo y unidad de predicción

Cada fila representa la demanda eléctrica diaria nacional publicada por REE. El objetivo es predecir la demanda del día `t` a un día vista. No se infiere causalidad ni se propone uso operativo por parte de REE.

## 2. Fuentes y trazabilidad

La demanda procede de REData mediante siete peticiones anuales entre 2019 y 2025. El clima procede de la Historical Weather API de Open-Meteo con reanálisis ERA5 diario para Madrid, Barcelona, Valencia, Sevilla y Bilbao. Se emplea `python-holidays` para festivos nacionales de España.

Cada extracción guarda URLs, rango, recuentos, timestamp y SHA-256. Los JSON originales no contienen datos personales ni credenciales.

## 3. Validación de calidad

Antes de transformar se comprueban:

- columnas requeridas, valores numéricos y demanda positiva;
- fechas únicas y continuidad diaria;
- respuesta meteorológica completa para cada ciudad y fecha;
- pesos de ciudades que suman uno para cada día;
- cobertura exacta entre demanda y proxy meteorológico.

El snapshot publicado contiene 2.557 días de demanda y 12.785 observaciones clima-ciudad; no hay días ausentes.

## 4. Proxy climático

El proyecto promedia con igual peso las cinco ciudades configuradas. Conserva temperatura media, máxima, mínima, rango térmico, dispersión entre ciudades, precipitación, viento máximo y radiación. Se derivan grados-día de calefacción `max(18 - temperatura, 0)` y refrigeración `max(temperatura - 22, 0)`.

Este proxy no pretende representar una media oficial nacional ni una ponderación por consumo. Es una decisión explícita y reproducible para estudiar señales climáticas de amplia cobertura territorial.

## 5. Calendario y disponibilidad

Las variables de calendario son conocidas antes de `t`: día de semana, mes, Fourier anual y semanal, festivo nacional, día anterior/posterior a festivo y puente entre festivo y fin de semana.

Los lags `1/7/14/28/364`, las medias móviles `7/28/56` y desviaciones móviles se desplazan un día: solo usan demanda observada hasta `t-1`. La tarea se denomina one-step-ahead porque en cada día de evaluación se permite actualizar esos lags con la demanda observada previa.

## 6. Hipótesis meteorológica

El candidato climático usa el reanálisis ERA5 realizado del día objetivo. En un despliegue se necesitaría un forecast meteorológico disponible antes de `t`, con su propio error. Por tanto, su resultado estima el valor potencial de una señal meteorológica, no el rendimiento exacto de un sistema productivo.

## 7. Validación temporal y selección

El pipeline define 12 orígenes mensuales de 2024. En cada origen entrena con todo el pasado disponible y evalúa los 28 días siguientes. Ningún dato de 2025 participa en selección ni tuning.

| Candidato | Variables |
|---|---:|
| Seasonal naive 7d | Demanda t-7 |
| Seasonal naive 364d | Demanda t-364 |
| Ridge autoregressive | Calendario y 22 señales históricas |
| HGB autoregressive | Calendario y 22 señales históricas |
| HGB weather informed | Las anteriores y 10 señales de clima |

La selección minimiza el MAE medio de las 12 ventanas. El HGB meteorológico se seleccionó con MAE medio 9.997 MWh y desviación estándar 3.111 MWh.

## 8. Calibración de sesgo e intervalos

Después de seleccionar el modelo se extraen 336 residuos fuera de muestra de 2024. Su mediana determina una corrección de sesgo de +1.549 MWh aplicada a las predicciones de 2025. La corrección se fija antes de observar el test.

Los intervalos simétricos conformales usan los cuantiles con corrección finita de los residuos absolutos corregidos. En 2025 alcanzan 72,1% de cobertura para el objetivo 80% y 94,0% para el objetivo 95%. La primera desviación se conserva como evidencia de posible cambio de régimen, no se recalibra con el test.

## 9. Métricas y contraste

Se informa MAE, RMSE, sMAPE, WAPE, sesgo y MASE respecto al baseline 7d. La comparación con el baseline semanal incluye:

- Diebold-Mariano sobre pérdida absoluta con varianza de largo plazo Newey-West de 7 días;
- bootstrap de bloques semanales con 1.000 remuestras para el intervalo de mejora de MAE.

En el test temporal el modelo seleccionado obtiene MAE 12.546 MWh, frente a 32.816 MWh del baseline. La ventaja bootstrap es 20.270 MWh, IC 95% [16.044, 24.506].

## 10. Diagnóstico y monitorización

Los informes incluyen error por mes, día de semana y régimen térmico; importancia por permutación post-hoc sobre el test; cobertura de intervalos; y consultas DuckDB sobre temperatura, grados-día y festivos. La importancia no demuestra causalidad ni se utiliza para seleccionar el modelo.

## 11. Reproducibilidad

Las dependencias numéricas se fijan a las versiones usadas para entrenar. El bundle guarda versiones de Python, scikit-learn, joblib, NumPy y pandas. Cambiar ese stack requiere reentrenar el modelo y no reutilizar el artefacto serializado.

## 12. Limitaciones

- No hay backtest con previsiones meteorológicas históricas emitidas en tiempo real.
- Las fechas de REE pueden revisarse y los snapshots futuros pueden variar.
- No se incorporan precios, actividad económica, generación, restricciones de red ni festivos locales.
- La evaluación es de un día vista y no mide degradación en forecast recursivo de largo horizonte.
