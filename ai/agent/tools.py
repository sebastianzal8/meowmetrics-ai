"""LangChain tools exposed to the ReAct agent.

Each tool is deliberately coarse-grained enough for a 3b model to call reliably:
query SQL, search docs, or list cats — nothing else.
"""
from __future__ import annotations

import re

from langchain_core.tools import tool

from ai.config import CONFIG
from ai.data_access.gold_client import BI_TABLE
from ai.data_access.sandbox import run_select_in_sandbox
from ai.data_access.sql_guard import SQLGuardError, sanitize_sql
from ai.rag.retriever import get_doc_index


def _warehouse_mode() -> bool:
    return CONFIG.has_snowflake


# --- Anti-infinite-loop guard -------------------------------------------------
# Small local models (llama3.2:3b) tend to retry the exact same failing SQL
# forever. We remember each identical failed query; after the second identical
# failure the tool returns an explicit STOP signal instead of the raw error.
_LAST_FAILURES: dict[str, int] = {}


def write_sql_key(sql: str) -> str:
    """Stable key for the failure ledger (normalized whitespace)."""
    return " ".join(sql.split()).lower()


def _note_failure(sql: str, original_msg: str) -> str:
    key = write_sql_key(sql)
    attempts = _LAST_FAILURES.get(key, 0) + 1
    _LAST_FAILURES[key] = attempts
    if attempts >= 2:
        return (
            "IDENTICAL QUERY ALREADY FAILED TWICE. STOP CALLING TOOLS. "
            "Either answer from the observations you already have, or state "
            "clearly that the question cannot be answered with the available table."
        )
    return original_msg


def _reset_failures() -> None:
    """Clears the failure ledger (used between agent invocations/tests)."""
    _LAST_FAILURES.clear()


def _execute_select(sql: str) -> list[tuple]:
    """Runs guard-approved SQL against Snowflake (prod) or the DuckDB sandbox (mock)."""
    hardened, _ = sanitize_sql(sql, max_rows=CONFIG.max_rows_per_query)
    if _warehouse_mode():
        from ai.data_access.gold_client import run_select
        _cols, rows = run_select(hardened)
        return rows
    return run_select_in_sandbox(hardened)


def _rows_to_markdown(rows: list[tuple], max_cell_chars: int = 40) -> str:
    """Compact markdown rendering so small context windows don't explode."""
    if not rows:
        return "0 rows returned."
    header = " | ".join(str(c) for c in rows[0])
    sep = "-" * len(header)
    body_lines = [
        " | ".join(str(v)[:max_cell_chars] for v in row)
        for row in rows[1:]
    ]
    return "\n".join([header, sep] + body_lines)


@tool
def query_gold_layer(sql: str) -> str:
    """Execute a READ-ONLY SELECT against MEOW_DB.ANALYTICS.BI_TABLEAU_VIEWS.
    One single SELECT statement per call. Never DML/DDL. Tables available:
    BI_TABLEAU_VIEWS (denormalized daily per-cat telemetry + health score).
    You may write the table name qualified (MEOW_DB.ANALYTICS.BI_TABLEAU_VIEWS) or bare."""
    try:
        rows = _execute_select(sql)
        _LAST_FAILURES.pop(write_sql_key(sql), None)
        return _rows_to_markdown(rows)
    except SQLGuardError as exc:
        return _note_failure(
            sql,
            f"SQL REJECTED by guardrail: {exc}\nRewrite as a single SELECT … LIMIT n.",
        )
    except Exception as exc:  # Snowflake/duckdb execution errors surface to the agent
        return _note_failure(
            sql,
            f"SQL EXECUTION ERROR: {exc}\nFix the SQL and retry.",
        )


@tool
def search_project_docs(query: str) -> str:
    """Semantic search over MeowMetrics internal engineering docs (README,
    dbt models & tests, Snowflake DDL, Databricks notebooks, Airflow DAG).
    Returns cited chunks with their source file path."""
    index = get_doc_index()
    context = index.format_context(query, k=CONFIG.retrieval_k)
    return f"Retrieved documentation context:\n\n{context}"


@tool
def get_cat_directory() -> str:
    """List every cat in the registry with cat_id, cat_name, breed and
    weight_category. Use it to resolve names/IDs before other SQL."""
    rows = _execute_select(
        f"SELECT cat_id, cat_name, breed, weight_category, age_years FROM {BI_TABLE} "
        "GROUP BY cat_id, cat_name, breed, weight_category, age_years LIMIT 50"
    )
    return _rows_to_markdown(rows)


ALL_TOOLS = [query_gold_layer, search_project_docs, get_cat_directory]

_SOURCE_RE = re.compile(r"source file: ([^\n\r]+)", re.IGNORECASE)


def extract_sources(tool_messages_content: list[str]) -> list[str]:
    """Parse unique file paths the agent really saw in doc observations."""
    sources, seen = [], set()
    for content in tool_messages_content:
        for match in _SOURCE_RE.finditer(content):
            path = match.group(1)
            if path not in seen:
                seen.add(path)
                sources.append(path)
    return sources
