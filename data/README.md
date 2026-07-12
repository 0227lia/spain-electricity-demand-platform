# Datos

El pipeline usa dos fuentes públicas y no necesita claves:

- REData de Red Eléctrica para demanda diaria nacional.
- Historical Weather API de Open-Meteo, modelo ERA5, para cinco ciudades españolas.

## Raw

- `raw/ree_demand_daily.json`: snapshot consolidado de REE.
- `raw/source_manifest.json`: URLs, ventanas, recuentos y hash de demanda.
- `raw/open_meteo_weather.json`: respuestas ERA5 por ciudad.
- `raw/weather_manifest.json`: URLs, ciudades, recuentos y hash de clima.

## Processed

- `processed/daily_demand.csv`: demanda validada y calendario nacional.
- `processed/city_weather.csv`: clima diario en cada ciudad.
- `processed/national_weather_proxy.csv`: proxy climático de igual peso.
- `processed/model_features.csv`: tabla final de entrenamiento y forecasting.

El CSV fuente no incluye información personal. La fecha de extracción y los hashes permiten detectar revisiones de las APIs. El periodo se configura en `.env.example` y `.env` nunca se versiona.
