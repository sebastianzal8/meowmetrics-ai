-- =====================================================================
-- Snowflake COPY INTO Script for MeowMetrics Ingestion
-- Demonstrates ingestion of Parquet (from Databricks) and CSV datasets
-- into raw staging schemas.
-- =====================================================================

USE ROLE MEOW_ADMIN_ROLE;
USE DATABASE MEOW_DB;
USE WAREHOUSE MEOW_WH;

-- 1. LOAD PARQUET SMART-COLLAR TELEMETRY FROM EXTERNAL STAGE
-- Since the Databricks output is formatted as Parquet, we parse individual
-- fields dynamically from the `$1` variant structure during ingestion.
COPY INTO RAW_STAGE.STG_RAW_COLLAR_TELEMETRY
FROM (
    SELECT -- noqa: ST06
        $1 AS SOURCE_VARIANT,
        $1:collar_id::VARCHAR AS COLLAR_ID,
        $1:cat_id::VARCHAR AS CAT_ID,
        $1:parsed_reading_time::TIMESTAMP AS READING_TIME,
        $1:heart_rate_bpm::INT AS HEART_RATE_BPM,
        $1:activity_level::VARCHAR AS ACTIVITY_LEVEL,
        $1:steps_count::INT AS STEPS_COUNT,
        $1:purr_frequency_hz::FLOAT AS PURR_FREQUENCY_HZ,
        $1:ambient_temp_c::FLOAT AS AMBIENT_TEMP_C,
        $1:battery_level_pct::INT AS BATTERY_LEVEL_PCT,
        CURRENT_TIMESTAMP()::TIMESTAMP AS INGESTED_AT,
        METADATA$FILENAME::VARCHAR AS SOURCE_FILE
    FROM @RAW_STAGE.MEOW_EXT_STAGE_TELEMETRY
)
FILE_FORMAT = (FORMAT_NAME = 'RAW_STAGE.PARQUET_FILE_FORMAT')
ON_ERROR = 'CONTINUE';


-- 2. LOAD CSV BREED AND REGISTRATION DATA FROM EXTERNAL STAGE
-- Simple columnar copy, mapping fields sequentially from the CSV file.
COPY INTO RAW_STAGE.STG_RAW_CAT_REGISTRY
FROM @RAW_STAGE.MEOW_EXT_STAGE_REGISTRY
FILE_FORMAT = (FORMAT_NAME = 'RAW_STAGE.CSV_FILE_FORMAT')
ON_ERROR = 'ABORT_STATEMENT'; -- Abort if reference data metadata is corrupted


-- 3. AUDIT RECENT LOADS
-- Query metadata details to confirm load progress
SELECT
    TABLE_NAME,
    ROW_COUNT,
    LAST_ALTERED
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'RAW_STAGE';
