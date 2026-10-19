"""Unit tests for the agent-facing SQL guard. Pure python + sqlglot + pandas + duckdb."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

from ai.data_access.sql_guard import SQLGuardError, sanitize_sql  # noqa: E402

FORBIDDEN_CASES = [
    "SELECT * FROM x; DROP TABLE dim_cats",
    "DELETE FROM BI_TABLEAU_VIEWS WHERE 1=1",
    "INSERT INTO BI_TABLEAU_VIEWS VALUES (1)",
    "UPDATE BI_TABLEAU_VIEWS SET cat_name = 'hacker'",
    "TRUNCATE TABLE BI_TABLEAU_VIEWS",
    "GRANT ALL ON MEOW_DB TO ROLE attacker",
    "SELECT 1 UNION SELECT 2; COPY @stage INTO @xx",
    "CREATE TABLE evil (a INT)",
    "MERGE INTO BI_TABLEAU_VIEWS USING s ON 1=1 WHEN MATCHED THEN UPDATE SET a=1",
]

SELECT_CASES = [
    ("SELECT cat_name FROM BI_TABLEAU_VIEWS LIMIT 5", True),
    ("SELECT cat_name FROM BI_TABLEAU_VIEWS", True),  # LIMIT gets injected
    ("WITH t AS (SELECT 1 AS a) SELECT * FROM t", True),
]


@pytest.mark.parametrize("bad", FORBIDDEN_CASES)
def test_rejects_writes_and_ddl(bad: str):
    with pytest.raises(SQLGuardError):
        sanitize_sql(bad)


@pytest.mark.parametrize("sql,ok", SELECT_CASES)
def test_allows_reads(sql: str, ok: bool):
    hardened, limit = sanitize_sql(sql)
    assert ok
    assert hardened.upper().count("SELECT") >= 1
    assert limit is not None


def test_capacity_is_capped():
    _, limit = sanitize_sql("SELECT cat_name FROM BI_TABLEAU_VIEWS LIMIT 100000")
    assert limit == 200  # configured cap


def test_fence_stripping():
    raw = "```sql\nSELECT cat_name FROM BI_TABLEAU_VIEWS LIMIT 10\n```"
    hardened, _ = sanitize_sql(raw)
    assert "```" not in hardened


def test_multi_statement_rejected():
    with pytest.raises(SQLGuardError):
        sanitize_sql("SELECT 1; SELECT 2")


def test_empty_rejected():
    with pytest.raises(SQLGuardError):
        sanitize_sql("")
