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
