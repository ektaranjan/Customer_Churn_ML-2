-- churn_analysis.sql
-- Run AFTER create_database.sql and after importing both CSVs.
-- USE customer_churn_db;

-- 1. Total customers
SELECT COUNT(*) AS total_customers FROM customer_churn;

-- 2. Total churned customers
SELECT COUNT(*) AS churned_customers FROM customer_churn WHERE Churn = 1;

-- 3. Churn rate (%)
SELECT ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct FROM customer_churn;

-- 4. Churn by contract
SELECT Contract,
       COUNT(*) AS total,
       SUM(Churn) AS churned,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 5. Churn by payment method
SELECT PaymentMethod,
       COUNT(*) AS total,
       SUM(Churn) AS churned,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 6. Average monthly charges by churn status
SELECT Churn, ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customer_churn
GROUP BY Churn;

-- 7. Average tenure by churn status
SELECT Churn, ROUND(AVG(tenure), 2) AS avg_tenure_months
FROM customer_churn
GROUP BY Churn;

-- 8. High-value customers (monthly charges above overall average + 1 std dev)
SELECT *
FROM customer_churn
WHERE MonthlyCharges > (SELECT AVG(MonthlyCharges) + STDDEV(MonthlyCharges) FROM customer_churn)
ORDER BY MonthlyCharges DESC;

-- 9. Potential revenue at risk (sum of monthly charges for HIGH risk customers,
--    from the model's prediction table — this is POTENTIAL risk, not a
--    guaranteed loss)
SELECT ROUND(SUM(MonthlyCharges), 2) AS potential_monthly_revenue_at_risk
FROM churn_predictions
WHERE Risk_Level = 'HIGH';

-- 10. Churn by internet service
SELECT InternetService,
       COUNT(*) AS total,
       SUM(Churn) AS churned,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- 11. Churn by tech support
SELECT TechSupport,
       COUNT(*) AS total,
       SUM(Churn) AS churned,
       ROUND(100.0 * SUM(Churn) / COUNT(*), 2) AS churn_rate_pct
FROM customer_churn
GROUP BY TechSupport
ORDER BY churn_rate_pct DESC;

-- 12. Customers with high churn probability (from model predictions)
SELECT CustomerID, Churn_Probability, Risk_Level, MonthlyCharges, Tenure, Contract
FROM churn_predictions
WHERE Risk_Level = 'HIGH'
ORDER BY Churn_Probability DESC;
