# Databricks notebook source
# DBTITLE 1,Configuration & Setup
from pyspark.sql.functions import current_timestamp, input_file_name
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Define workspace paths (Simulated AWS S3 Mounts)
LANDING_ZONE_PATH = "dbfs:/mnt/meowmetrics-lakehouse/landing/collar_telemetry"
BRONZE_TABLE_PATH = "dbfs:/mnt/meowmetrics-lakehouse/bronze/collar_telemetry"
CHECKPOINT_PATH = "dbfs:/mnt/meowmetrics-lakehouse/checkpoints/bronze_collar_telemetry"

# DBTITLE 2,Define Schema for Smart Collar JSON Telemetry
# Smart collars send hourly IoT readings of activity index, purr frequency, sleeping patterns, ambient temp, and heart rate
collar_schema = StructType([
    StructField("collar_id", StringType(), False),
    StructField("cat_id", StringType(), False),
    StructField("reading_time", StringType(), False),
    StructField("heart_rate_bpm", IntegerType(), True),
    StructField("activity_level", StringType(), True),  # 'sleep', 'active', 'lazy'
    StructField("steps_count", IntegerType(), True),
    StructField("purr_frequency_hz", DoubleType(), True),
    StructField("ambient_temp_c", DoubleType(), True),
    StructField("battery_level_pct", IntegerType(), True)
])

# DBTITLE 3,Read Stream with Auto Loader
# Auto Loader efficiently processes new files arriving in cloud storage (S3/ADLS)
print(f"Initializing Auto Loader stream from: {LANDING_ZONE_PATH}")

raw_stream_df = (
    spark.readStream  # noqa: F821
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", f"{CHECKPOINT_PATH}/schema")
    .schema(collar_schema)
    .load(LANDING_ZONE_PATH)
)

# DBTITLE 4,Add Metadata Columns (Bronze Ingestion Auditing)
# Adding metadata (ingestion timestamp, source file path) helps debug pipeline issues
bronze_df = (
    raw_stream_df
    .withColumn("ingested_at", current_timestamp())
    .withColumn("source_file", input_file_name())
)

# DBTITLE 5,Write Stream to Delta Lake (Bronze Table)
# Delta Lake provides ACID transactions, scalability, and time travel features
print(f"Streaming write to Bronze Delta table at: {BRONZE_TABLE_PATH}")

query = (
    bronze_df.writeStream
    .format("delta")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .outputMode("append")
    .start(BRONZE_TABLE_PATH)
)

# In real Databricks execution, the stream would run continuously or with trigger-once logic
# query.awaitTermination()
print("Bronze stream ingestion pipeline initialized successfully.")
