# Metodologia

## Extraccion y calidad

El pipeline consulta la API publica de REData por ventanas anuales. Usa TLS verificado con el almacen de certificados del sistema, tres intentos acotados y un `User-Agent` identificable. Cada ejecucion guarda las URLs, fechas, numero de registros y hash SHA-256 del snapshot.

Antes de cargar el almacen se valida que la serie tenga las columnas esperadas, valores positivos, fechas unicas y continuidad diaria entre la primera y la ultima observacion. Los codigos de respuesta y los ficheros de log no contienen credenciales.

## Modelo de datos

DuckDB contiene:

- `fact_daily_demand`: demanda diaria en MWh y procedencia.
- `dim_date`: atributos de calendario para cada fecha.
- `mart_monthly_demand`: vista agregada mensual.

Las consultas analiticas estan versionadas en `sql/analytics.sql` y se exportan a `reports/sql/` en cada ejecucion.

## Forecast

La tarea es un forecast diario a un paso: para predecir el dia `t`, los lags y medias moviles terminan en `t-1`. Esto evita usar la demanda objetivo del propio dia. Se comparan:

- baseline estacional semanal: demanda de `t-7`;
- `HistGradientBoostingRegressor` con calendario, lags 1/7/14/28 y medias moviles de 7/28 dias.

La seleccion se hace por MAE medio en cuatro ventanas de backtesting de 28 dias durante 2024. El periodo de 2025 queda reservado como test temporal y no se usa para elegir modelo ni parametros.

## Limitaciones

- Es un forecast a un paso, no una prediccion recursiva de varias semanas sin observar nuevos datos.
- No incorpora meteorologia, festividades especificas, precios ni actividad economica.
- Los resultados dependen de la definicion de demanda expuesta por REE y de posibles revisiones de la fuente.
- Los modelos describen error predictivo en este periodo; no explican causalidad ni sustituyen planificacion del sistema electrico.
