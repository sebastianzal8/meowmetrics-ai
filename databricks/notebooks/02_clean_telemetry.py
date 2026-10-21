# Databricks notebook source
# DBTITLE 1,Imports & Configuration
from pyspark.sql.functions import (
    col,
    row_number,
    to_timestamp,
    when,
)
from pyspark.sql.window import Window

# Target directories
BRONZE_TABLE_PATH = "dbfs:/mnt/meowmetrics-lakehouse/bronze/collar_telemetry"
SILVER_TABLE_PATH = "dbfs:/mnt/meowmetrics-lakehouse/silver/collar_telemetry_cleaned"
CAT_REGISTRY_PATH = "dbfs:/mnt/meowmetrics-lakehouse/raw/cat_registry"  # Static profiles

# DBTITLE 2,Load Bronze Data and Static Metadata
print("Loading Bronze telemetry and Cat Breed Registry metadata...")

bronze_df = spark.read.format("delta").load(BRONZE_TABLE_PATH)  # noqa: F821
registry_df = spark.read.format("csv").option("header", "true").load(CAT_REGISTRY_PATH)  # noqa: F821

# DBTITLE 3,Data Cleaning and Quality Guardrails
# 1. Parse timestamps
# 2. Enforce logic limits (heart rate: cats rest around 120-140, active up to 220. <40 or >260 is considered sensor error)
# 3. Handle NULL battery rates by defaulting to -1 (indicating sensor anomaly)

cleaned_df = bronze_df \
    .withColumn("parsed_reading_time", to_timestamp(col("reading_time"), "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("heart_rate_bpm", when((col("heart_rate_bpm") >= 40) & (col("heart_rate_bpm") <= 250), col("heart_rate_bpm")).otherwise(None)) \
    .withColumn("battery_level_pct", when((col("battery_level_pct") >= 0) & (col("battery_level_pct") <= 100), col("battery_level_pct")).otherwise(-1)) \
    .withColumn("steps_count", when(col("steps_count") >= 0, col("steps_count")).otherwise(0))

# DBTITLE 4,De-duplication using Window Function
# Collars might send the same message twice due to network latency retries.
# We deduplicate on (collar_id, parsed_reading_time), keeping the most recently ingested record.

window_spec = Window.partitionBy("collar_id", "parsed_reading_time").orderBy(col("ingested_at").desc())

deduplicated_df = cleaned_df \
    .withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

# DBTITLE 5,Join Telemetry with Cat Breed Registry
# Enriches the collar telemetry with Cat details (name, breed, gender, birth date)
enriched_df = deduplicated_df.join(
    registry_df,
    on="cat_id",
    how="inner"
)

# DBTITLE 6,Write Silver Table
# Silver tables are clean, verified, and ready for analytics databases (like Snowflake)
print(f"Writing structured and cleaned data to Silver Delta Lake: {SILVER_TABLE_PATH}")

(
    enriched_df.write
    .format("delta")
    .mode("overwrite")  # In production, this can be append or merge based on time slices
    .option("overwriteSchema", "true")
    .save(SILVER_TABLE_PATH)
)

print("Silver table processing completed successfully.")
