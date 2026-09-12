-- create_database.sql
-- Run this first in MySQL Workbench (or `mysql -u root -p < create_database.sql`)

CREATE DATABASE IF NOT EXISTS customer_churn_db;
USE customer_churn_db;

-- Main customer table — matches the columns in
-- Dataset/processed/customer_churn_cleaned.csv
CREATE TABLE IF NOT EXISTS customer_churn (
    gender            VARCHAR(10),
    SeniorCitizen     INT,
    Partner           VARCHAR(5),
    Dependents        VARCHAR(5),
    tenure            INT,
    PhoneService      VARCHAR(5),
    MultipleLines     VARCHAR(20),
    InternetService   VARCHAR(20),
    OnlineSecurity    VARCHAR(20),
    OnlineBackup      VARCHAR(20),
    DeviceProtection  VARCHAR(20),
    TechSupport       VARCHAR(20),
    StreamingTV       VARCHAR(20),
    StreamingMovies   VARCHAR(20),
    Contract          VARCHAR(20),
    PaperlessBilling  VARCHAR(5),
    PaymentMethod     VARCHAR(30),
    MonthlyCharges    DECIMAL(8,2),
    TotalCharges      DECIMAL(10,2),
    Churn             INT
);

-- Predictions table — matches Dataset/processed/churn_predictions.csv
CREATE TABLE IF NOT EXISTS churn_predictions (
    CustomerID          VARCHAR(20),
    Churn                VARCHAR(5),
    Churn_Probability    DECIMAL(6,2),
    Risk_Level            VARCHAR(10),
    MonthlyCharges       DECIMAL(8,2),
    Tenure               INT,
    Contract             VARCHAR(20),
    TotalCharges         DECIMAL(10,2)
);

-- HOW TO IMPORT THE CSVs (MySQL Workbench):
-- 1. Right-click "Tables" under customer_churn_db -> "Table Data Import Wizard"
-- 2. Select Dataset/processed/customer_churn_cleaned.csv -> Next
-- 3. Choose "Use existing table" -> customer_churn -> Next -> Next -> Finish
-- 4. Repeat for Dataset/processed/churn_predictions.csv -> churn_predictions table
