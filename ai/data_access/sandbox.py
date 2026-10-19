"""In-memory SQL sandbox over the mock Gold layer (DuckDB).

Used when Snowflake credentials are absent (demo mode + CI evals) so that
text-to-SQL answers and eval scoring execute *real* SQL — no LLM, no network.

The sandbox mimics the warehouse layout (catalog MEOW_DB, schema ANALYTICS)
so the agent can use fully-qualified names identical to production.
"""
from __future__ import annotations

import duckdb

from ai.data_access.mock_data import build_mock_gold_data

SANDBOX_CATALOG = "MEOW_DB"
SANDBOX_SCHEMA = "ANALYTICS"

_STAGING_VIEW = "__MOCK_BI_TABLEAU_VIEWS__"

_SANDABOX_CONNECTION = None

_ATTACHER_SQL = f"""
ATTACH ':memory:' AS {SANDBOX_CATALOG};
CREATE SCHEMA {SANDBOX_CATALOG}.{SANDBOX_SCHEMA};
CREATE TABLE {SANDBOX_CATALOG}.{SANDBOX_SCHEMA}.BI_TABLEAU_VIEWS AS SELECT * FROM {_STAGING_VIEW};
-- Bare alias so simple queries without full qualification also work:
CREATE OR REPLACE VIEW BI_TABLEAU_VIEWS AS SELECT * FROM {_STAGING_VIEW};
"""


def get_sandbox():
    """Lazy singleton DuckDB connection with the mock Gold table registered."""
    global _SANDABOX_CONNECTION
    if _SANDABOX_CONNECTION is None:
        con = duckdb.connect(":memory:")
        con.register(_STAGING_VIEW, build_mock_gold_data())
        con.execute(_ATTACHER_SQL)
        _SANDABOX_CONNECTION = con
    return _SANDABOX_CONNECTION


def run_select_in_sandbox(sql: str) -> list[tuple]:
    """Execute a whitelisted SELECT against mock data. Returns raw rows."""
    con = get_sandbox()
    cur = con.execute(sql)
    return cur.fetchall()
