"""Read-only access to the Snowflake Gold layer (ANALYTICS schema).

Safety model:
- One lazily-created connection per process, opened per call batch and force-closed
  after every use (Snowflake connections are cheap to open).
- We never build SQL from user input here — callers pass guard-approved SQL only.
- A module-level deny list is re-checked at execution time as a final tripwire.
"""
from __future__ import annotations

import pandas as pd

from ai.config import CONFIG
from ai.data_access.sql_guard import SQLGuardError, sanitize_sql

_DENY_TOKENS = {"insert", "update", "delete", "merge", "drop", "alter", "create", "truncate",
                "grant", "revoke", "copy", "put", "get", "set", "use", "show", "describe", "explain"}

BI_TABLE = "BI_TABLEAU_VIEWS"


def _assert_safe(sql: str) -> str:
    if any(tok in sql.lower() for tok in _DENY_TOKENS):
        raise SQLGuardError("Tripwire: once approved, re-checked and denied.")
    hardened, _ = sanitize_sql(sql, max_rows=CONFIG.max_rows_per_query)
    return hardened


def connect_sql():
    """Return a closed-on-exit Snowflake connection context manager."""
    if not CONFIG.has_snowflake:
        raise RuntimeError("Snowflake credentials are not configured (SF_USER/SF_PASSWORD/SF_ACCOUNT).")

    import snowflake.connector

    ctx = snowflake.connector.connect(
        user=CONFIG.sf_user,
        password=CONFIG.sf_password,
        account=CONFIG.sf_account,
        warehouse=CONFIG.sf_warehouse,
        database=CONFIG.sf_database,
        schema=CONFIG.sf_schema,
    )
    return ctx


def run_select(sql: str) -> tuple[list[str], list[tuple]]:
    """Execute a guard-approved SELECT on Snowflake. Returns (columns, rows)."""
    safe_sql = _assert_safe(sql)
    ctx = connect_sql()
    try:
        cs = ctx.cursor()
        cs.execute(safe_sql)
        cols = [c[0].lower() for c in cs.description]
        rows = cs.fetchall()
        return cols, rows
    finally:
        cs.close()
        ctx.close()


def load_bi_tableau_views() -> pd.DataFrame:
    """Pull the denormalized BI table used by the dashboard and copilot."""
    cols, rows = run_select(f"SELECT * FROM {BI_TABLE}")
    return pd.DataFrame(rows, columns=cols)


def build_bi_table_schema_hint() -> str:
    """Introspect Snowflake into a compact CREATE TABLE hint for the LLM prompt."""
    ctx = connect_sql()
    try:
        cs = ctx.cursor()
        cs.execute(f"DESCRIBE TABLE MEOW_DB.{CONFIG.sf_schema}.{BI_TABLE}")
        schema = cs.fetchall()
    finally:
        cs.close()
        ctx.close()

    lines = [f"CREATE TABLE MEOW_DB.{CONFIG.sf_schema}.{BI_TABLE} ("]
    lines += [f"    {row[0]} {row[1]}," for row in schema[:-1]]
    lines.append(f"    {schema[-1][0]} {schema[-1][1]}")
    lines.append(");")
    return "\n".join(lines)
