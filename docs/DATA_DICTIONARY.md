# Diccionario de datos

## `daily_demand.csv`

| Campo | Descripcion |
|---|---|
| `date` | Fecha local de la observacion diaria. |
| `demand_mwh` | Demanda diaria en MWh tal como la expone el endpoint de REE. |
| `year`, `month`, `week` | Atributos de calendario. |
| `day_of_week` | Dia ISO de semana codificado de 0 (lunes) a 6 (domingo). |
| `is_weekend` | Indicador de sabado o domingo. |

## `model_features.csv`

Incluye las columnas anteriores y estas variables usadas para el forecast:

| Campo | Descripcion |
|---|---|
| `day_of_year_sin`, `day_of_year_cos` | Codificacion ciclica anual. |
| `lag_1`, `lag_7`, `lag_14`, `lag_28` | Demanda observada tantos dias antes. |
| `rolling_mean_7`, `rolling_mean_28` | Media de demanda historica, desplazada un dia. |

Todas las variables de lag y media movil usan observaciones anteriores a la fecha objetivo, por lo que no incluyen la demanda del mismo dia.
