-- =====================================================================
-- Snowflake Setup DDL Script for MeowMetrics
-- This script designs the database architecture, security rules, schemas,
-- external stages, and staging tables for the cat telemetry ingestion.
-- =====================================================================

-- 1. SECURITY & ACCESS CONTROL SETUP
USE ROLE SECURITYADMIN;

-- Create MeowMetrics service role
CREATE ROLE IF NOT EXISTS MEOW_ADMIN_ROLE;
GRANT ROLE MEOW_ADMIN_ROLE TO ROLE SYSADMIN;

-- 2. INFRASTRUCTURE & VIRTUAL WAREHOUSE CREATION
USE ROLE SYSADMIN;

CREATE WAREHOUSE IF NOT EXISTS MEOW_WH
WITH WAREHOUSE_SIZE = 'XSMALL'
AUTO_SUSPEND = 60
AUTO_RESUME = TRUE
INITIALLY_SUSPENDED = TRUE
COMMENT = 'Compute warehouse for MeowMetrics ELT pipelines';

-- Create Database
CREATE DATABASE IF NOT EXISTS MEOW_DB;

-- Transfer ownership of database and warehouse to custom role
GRANT OWNERSHIP ON DATABASE MEOW_DB TO ROLE MEOW_ADMIN_ROLE
COPY CURRENT GRANTS;
GRANT OWNERSHIP ON WAREHOUSE MEOW_WH TO ROLE MEOW_ADMIN_ROLE
COPY CURRENT GRANTS;

-- 3. SCHEMA & OBJECT CREATION
USE ROLE MEOW_ADMIN_ROLE;
USE DATABASE MEOW_DB;

-- Raw Stage Schema: holds raw data stages and loading tables
CREATE SCHEMA IF NOT EXISTS RAW_STAGE;

-- Analytics Schema: holds dbt models, stars, facts, and marts
CREATE SCHEMA IF NOT EXISTS ANALYTICS;

-- 4. FILE FORMATS DEFINITIONS
USE SCHEMA RAW_STAGE;

-- CSV File Format for Cat Breed/Registry CSV files
CREATE OR REPLACE FILE FORMAT CSV_FILE_FORMAT
TYPE = 'CSV'
FIELD_DELIMITER = ','
SKIP_HEADER = 1
NULL_IF = ('NULL', 'null', '')
EMPTY_FIELD_AS_NULL = TRUE
ERROR_ON_COLUMN_MISMATCH = FALSE; -- noqa: PRS

-- Parquet File Format for Databricks Silver outputs
CREATE OR REPLACE FILE FORMAT PARQUET_FILE_FORMAT
TYPE = 'PARQUET'
COMPRESSION = 'SNAPPY';

-- 5. EXTERNAL STAGES CONFIGURATION
-- Points to the storage locations where Databricks exports the cleaned data.
-- (Credentials should be configured using Snowflake Storage Integrations in production).
CREATE OR REPLACE STAGE MEOW_EXT_STAGE_TELEMETRY
    URL = 's3://meowmetrics-lakehouse/silver/collar_telemetry_cleaned/'
    FILE_FORMAT = PARQUET_FILE_FORMAT;

CREATE OR REPLACE STAGE MEOW_EXT_STAGE_REGISTRY
    URL = 's3://meowmetrics-lakehouse/raw/cat_registry/'
    FILE_FORMAT = CSV_FILE_FORMAT;

-- 6. STAGING TABLES FOR DATA LOADING
-- Raw table for IoT smart-collar telemetry (stores structured elements)
CREATE OR REPLACE TABLE RAW_STAGE.STG_RAW_COLLAR_TELEMETRY (
    collar_id VARCHAR(50),
    cat_id VARCHAR(50),
    reading_time TIMESTAMP,
    heart_rate_bpm INT,
    activity_level VARCHAR(30),
    steps_count INT,
    purr_frequency_hz FLOAT,
    ambient_temp_c FLOAT,
    battery_level_pct INT,
    ingested_at TIMESTAMP,
    source_file VARCHAR(255)
) COMMENT = 'Staging table for Databricks cleaned telemetry';

-- Raw table for static cat breed registry metadata
CREATE OR REPLACE TABLE RAW_STAGE.STG_RAW_CAT_REGISTRY (
    cat_id VARCHAR(50),
    name VARCHAR(100),
    breed VARCHAR(100),
    gender VARCHAR(10),
    birth_date DATE,
    weight_lbs FLOAT,
    owner_id VARCHAR(50),
    registered_at DATE
) COMMENT = 'Staging table for static registry records';

-- Provide usage permissions to dbt developer environment
GRANT USAGE ON WAREHOUSE MEOW_WH TO ROLE MEOW_ADMIN_ROLE;
GRANT ALL PRIVILEGES ON DATABASE MEOW_DB TO ROLE MEOW_ADMIN_ROLE;
GRANT ALL PRIVILEGES ON ALL SCHEMAS IN DATABASE MEOW_DB TO ROLE MEOW_ADMIN_ROLE;

SELECT 'Snowflake setup DDL executed successfully' AS status;
