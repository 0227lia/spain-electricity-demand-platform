-- name: monthly_demand
SELECT
    date_trunc('month', fact.date) AS month,
    SUM(fact.demand_mwh) AS total_demand_mwh,
    AVG(fact.demand_mwh) AS avg_daily_demand_mwh,
    MAX(fact.demand_mwh) AS max_daily_demand_mwh
FROM fact_daily_demand AS fact
GROUP BY 1
ORDER BY 1;

-- name: weekday_profile
SELECT
    date_part('dow', fact.date) AS day_of_week,
    AVG(fact.demand_mwh) AS avg_demand_mwh,
    MIN(fact.demand_mwh) AS min_demand_mwh,
    MAX(fact.demand_mwh) AS max_demand_mwh,
    COUNT(*) AS days
FROM fact_daily_demand AS fact
GROUP BY 1
ORDER BY 1;

-- name: annual_demand
SELECT
    date_part('year', fact.date) AS year,
    SUM(fact.demand_mwh) AS total_demand_mwh,
    AVG(fact.demand_mwh) AS avg_daily_demand_mwh,
    MAX(fact.demand_mwh) AS peak_daily_demand_mwh
FROM fact_daily_demand AS fact
GROUP BY 1
ORDER BY 1;

-- name: peak_days
SELECT
    fact.date,
    fact.demand_mwh,
    date_part('year', fact.date) AS year,
    date_part('month', fact.date) AS month,
    date_part('dow', fact.date) AS day_of_week
FROM fact_daily_demand AS fact
ORDER BY fact.demand_mwh DESC
LIMIT 20;

-- name: monthly_weather
SELECT
    date_trunc('month', date) AS month,
    AVG(temperature_mean_c) AS avg_temperature_c,
    SUM(heating_degree_days) AS heating_degree_days,
    SUM(cooling_degree_days) AS cooling_degree_days,
    SUM(precipitation_mm) AS precipitation_mm,
    AVG(wind_speed_max_kmh) AS avg_max_wind_kmh
FROM fact_daily_weather_proxy
GROUP BY 1
ORDER BY 1;

-- name: temperature_demand_profile
SELECT
    CASE
        WHEN temperature_mean_c < 8 THEN 'cold_below_8c'
        WHEN temperature_mean_c < 16 THEN 'cool_8_to_16c'
        WHEN temperature_mean_c < 24 THEN 'mild_16_to_24c'
        ELSE 'hot_24c_plus'
    END AS temperature_regime,
    COUNT(*) AS days,
    AVG(temperature_mean_c) AS avg_temperature_c,
    AVG(demand_mwh) AS avg_demand_mwh,
    MIN(demand_mwh) AS min_demand_mwh,
    MAX(demand_mwh) AS max_demand_mwh
FROM mart_daily_demand_weather
GROUP BY 1
ORDER BY avg_temperature_c;

-- name: calendar_demand_profile
SELECT
    CASE
        WHEN is_holiday THEN 'holiday'
        WHEN is_bridge_day THEN 'bridge_day'
        ELSE 'regular_day'
    END AS calendar_regime,
    COUNT(*) AS days,
    AVG(demand_mwh) AS avg_demand_mwh,
    MIN(demand_mwh) AS min_demand_mwh,
    MAX(demand_mwh) AS max_demand_mwh
FROM mart_daily_demand_weather
GROUP BY 1
ORDER BY avg_demand_mwh;

-- name: degree_day_sensitivity
SELECT
    CAST(FLOOR(temperature_mean_c / 2) * 2 AS INTEGER) AS temperature_bin_c,
    COUNT(*) AS days,
    AVG(demand_mwh) AS avg_demand_mwh,
    AVG(heating_degree_days) AS avg_heating_degree_days,
    AVG(cooling_degree_days) AS avg_cooling_degree_days
FROM mart_daily_demand_weather
GROUP BY 1
HAVING COUNT(*) >= 10
ORDER BY 1;
