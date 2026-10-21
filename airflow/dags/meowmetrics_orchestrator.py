from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator
from airflow.providers.databricks.operators.databricks import (
    DatabricksSubmitRunOperator,
)
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.utils.task_group import TaskGroup

from airflow import DAG

# Default arguments for the DAG (best practice for production pipelines)
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 1),
    'email': ['alerts@meowmetrics-health.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG Schedule
with DAG(
    'meowmetrics_data_pipeline',
    default_args=default_args,
    description='Orchestrates the Databricks, Snowflake, and dbt pipeline for Cat IoT Smart-Collar Telemetry',
    schedule_interval='@daily',
    catchup=False,
    tags=['lakehouse', 'telemetry', 'snowflake', 'dbt'],
) as dag:

    # 1. ORCHESTRATE DATABRICKS PYSPARK WORKLOADS
    # Uses task group to organize Databricks notebook execution
    with TaskGroup("databricks_processing") as databricks_group:
        
        # Ingest raw files using Auto Loader stream
        task_ingest_telemetry = DatabricksSubmitRunOperator(
            task_id='ingest_bronze_telemetry',
            json={
                'notebook_task': {
                    'notebook_path': '/Repos/production/meowmetrics/databricks/notebooks/01_ingest_telemetry',
                },
                'new_cluster': {
                    'spark_version': '13.3.x-scala2.12',
                    'node_type_id': 'i3.xlarge',
                    'num_workers': 2,
                }
            }
        )

        # De-duplicate, clean, and output Silver Parquet datasets
        task_clean_telemetry = DatabricksSubmitRunOperator(
            task_id='clean_silver_telemetry',
            json={
                'notebook_task': {
                    'notebook_path': '/Repos/production/meowmetrics/databricks/notebooks/02_clean_telemetry',
                },
                'existing_cluster_id': '0519-123456-pool8'  # Uses warm cluster to save spin-up time
            }
        )

        task_ingest_telemetry >> task_clean_telemetry

    # 2. INGEST DATA INTO SNOWFLAKE RAW TABLES
    # Executes SQL COPY INTO instructions to load the Parquet and CSV files into Snowflake
    with TaskGroup("snowflake_loading") as snowflake_group:
        
        task_copy_telemetry = SnowflakeOperator(
            task_id='copy_telemetry_to_staging',
            sql='snowflake/copy_into_staging.sql',
            snowflake_conn_id='snowflake_meow_connection',
            warehouse='MEOW_WH',
            database='MEOW_DB',
            schema='RAW_STAGE'
        )

    # 3. RUN TRANSFORMATION MODELS USING dbt
    # Executes dbt run and dbt test to build Gold-layer stars and validate schema checks
    with TaskGroup("dbt_modeling") as dbt_group:
        
        task_dbt_deps = BashOperator(
            task_id='dbt_install_dependencies',
            bash_command='cd /opt/airflow/dbt_project && dbt deps'
        )

        task_dbt_run = BashOperator(
            task_id='dbt_run_models',
            bash_command='cd /opt/airflow/dbt_project && dbt run --profiles-dir .'
        )

        task_dbt_test = BashOperator(
            task_id='dbt_test_models',
            bash_command='cd /opt/airflow/dbt_project && dbt test --profiles-dir .'
        )

        task_dbt_deps >> task_dbt_run >> task_dbt_test

    # 4. LLM DAILY DIGEST (optional — runs after dbt completes)
    # Generates per-cat natural-language health digests from Gold-layer data
    with TaskGroup("llm_digest") as digest_group:
        
        task_llm_digest = BashOperator(
            task_id='generate_daily_digests',
            bash_command='cd /opt/airflow && python -m ai.pipeline.daily_digest',
            env={
                'SF_USER': '{{ var.value.sf_user }}',
                'SF_PASSWORD': '{{ var.value.sf_password }}',
                'SF_ACCOUNT': '{{ var.value.sf_account }}',
                'SF_WAREHOUSE': 'MEOW_WH',
                'SF_DATABASE': 'MEOW_DB',
                'SF_SCHEMA': 'ANALYTICS',
                'OLLAMA_BASE_URL': 'http://host.docker.internal:11434',
                'AI_CHAT_MODEL': 'llama3.2:3b',
            }
        )

    # Define Top-Level Dependencies
    databricks_group >> snowflake_group >> dbt_group >> digest_group
