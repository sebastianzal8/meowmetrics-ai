"""LangChain tools exposed to the ReAct agent.

Each tool is deliberately coarse-grained enough for a 3b model to call reliably:
query SQL, search docs, list cats, or explain a metric — nothing else.
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache

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
    # Auto-fix common small-model SQL errors
    sql = _fix_common_sql_errors(sql)
    hardened, _ = sanitize_sql(sql, max_rows=CONFIG.max_rows_per_query)
    if _warehouse_mode():
        from ai.data_access.gold_client import run_select
        _cols, rows = run_select(hardened)
        return rows
    return run_select_in_sandbox(hardened)


@lru_cache(maxsize=1)
def _bi_columns() -> tuple[str, ...]:
    """Exact column names of the Gold table (mock mode; empty when on Snowflake)."""
    if _warehouse_mode():
        return ()
    from ai.data_access.sandbox import get_sandbox

    rows = get_sandbox().execute(f"DESCRIBE SELECT * FROM {BI_TABLE}").fetchall()
    return tuple(str(row[0]) for row in rows)


_COLUMN_ERROR_RE = re.compile(r'column\s+"([^"]+)"\s+not found', re.IGNORECASE)


def _friendly_sql_error(exc: Exception) -> str:
    """Turn a database exception into a short observation the model can act on.

    Small models invent column names (``average_heart_rate_bpm`` instead of
    ``avg_heart_rate_bpm``). Surfacing the real schema plus a closest match lets
    the ReAct loop repair itself instead of crashing the caller.
    """
    detail = " ".join(str(exc).split())
    if len(detail) > 400:
        detail = detail[:400] + "…"

    hint = ""
    missing = _COLUMN_ERROR_RE.search(detail)
    columns = _bi_columns()
    if missing and columns:
        bad = missing.group(1)
        close = difflib.get_close_matches(
            bad.lower(), [c.lower() for c in columns], n=1, cutoff=0.6
        )
        hint = "\nHint: " + (f"did you mean `{close[0]}`? " if close else "")
        hint += "Use EXACT column names: " + ", ".join(columns)
    return f"SQL EXECUTION ERROR: {detail}{hint}\nFix the SQL and retry."


def _fix_common_sql_errors(sql: str) -> str:
    """Auto-correct frequent mistakes from small models."""
    # Replace smart quotes and various Unicode quote characters with straight quotes
    # U+2018/U+2019: left/right single quotation marks
    # U+201C/U+201D: left/right double quotation marks
    # U+FF02: fullwidth quotation mark
    # U+201E/U+201F: double low-9/double high-reversed-9 quotation marks
    sql = (sql
        .replace('\u2018', "'")   # left single quotation mark
        .replace('\u2019', "'")   # right single quotation mark
        .replace('\u201c', "'")   # left double quotation mark
        .replace('\u201d', "'")   # right double quotation mark
        .replace('\u2013', "'")   # en dash (small models use it as a quote)
        .replace('\u2014', "'")   # em dash (same)
        .replace('\u201a', "'")   # single low-9 quotation mark
        .replace('\u201b', "'")   # single high-reversed-9 quotation mark
        .replace('\u2032', "'")   # prime
        .replace('\u2033', "'")   # double prime
        .replace('\u201e', "'")   # double low-9 quotation mark
        .replace('\u201f', "'")   # double high-reversed-9 quotation mark
        .replace('\uff02', "'")   # fullwidth quotation mark
        # Also handle replacement character (U+FFFD) that may appear when decoding fails
        .replace('\ufffd', "'")
    )
    
    # Replace "YYYY-MM-DD" with 'YYYY-MM-DD' (date literals)
    sql = re.sub(r'"(\d{4}-\d{2}-\d{2})"', r"'\1'", sql)
    
    # Replace "CATxxx" with 'CATxxx' (cat_id literals)
    sql = re.sub(r'"(CAT\d+)"', r"'\1'", sql, flags=re.IGNORECASE)
    
    # AGGRESSIVE: Replace ALL double-quoted strings that look like VALUES (not identifiers)
    # In SQL, double quotes are for identifiers, single quotes for string literals.
    # Any double-quoted string in a value context (after =, <, >, IN, LIKE, etc.) should be single quotes.
    
    # Pattern 1: = "value" or > "value" or < "value" or >= "value" or <= "value" or <> "value" or != "value"
    sql = re.sub(r'(=\s*|>\s*|<\s*|>=\s*|<=\s*|<>\s*|!=\s*)"([^"]+)"', r"\1'\2'", sql)
    
    # Pattern 2: IN ("val1", "val2", ...)
    def _fix_in_clause(match):
        prefix = match.group(1)
        values = match.group(2)
        # Replace all double quotes with single quotes in the IN list
        fixed_values = re.sub(r'"([^"]+)"', r"'\1'", values)
        return f"{prefix}{fixed_values})"
    
    sql = re.sub(r'(IN\s*\()\s*([^)]+)\s*\)', _fix_in_clause, sql)
    
    # Pattern 3: BETWEEN "val1" AND "val2"
    sql = re.sub(r'BETWEEN\s+"([^"]+)"\s+AND\s+"([^"]+)"', r"BETWEEN '\1' AND '\2'", sql, flags=re.IGNORECASE)
    
    # Pattern 4: LIKE "pattern"
    sql = re.sub(r'(LIKE\s+)"([^"]+)"', r"\1'\2'", sql, flags=re.IGNORECASE)
    
    return sql


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
    except Exception as exc:  # noqa: BLE001 - a tool must return errors, never crash the graph
        return _note_failure(sql, _friendly_sql_error(exc))


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
    """List the identity fields of every cat: cat_id, cat_name, breed and
    age_years. Use it ONLY to resolve names/IDs before writing SQL; it is not
    a source of analytics answers (weight, health, steps)."""
    try:
        rows = _execute_select(
            f"SELECT cat_id, cat_name, breed, age_years FROM {BI_TABLE} "
            "GROUP BY cat_id, cat_name, breed, age_years ORDER BY cat_id LIMIT 50"
        )
    except Exception as exc:  # noqa: BLE001 - a tool must return errors, never crash the graph
        return _friendly_sql_error(exc)
    return _rows_to_markdown(rows)


@tool
def explain_metric(metric_name: str) -> str:
    """Return the exact formula and logic for a named health metric.

    Use this when the user asks "how is X calculated?" or "what is the formula for X?".
    The metric_name should be one of: 'health_score', 'health_status', 'daily_feline_health_score'.

    Returns the dbt model logic with CASE statements, thresholds, and penalties.
    """
    metric_name = metric_name.lower().strip()

    if metric_name in ('health_score', 'health_status', 'daily_feline_health_score', 'daily_cat_health'):
        return (
            "MEOWMETRICS DAILY FELINE HEALTH SCORE FORMULA (from dbt daily_cat_health.sql):\n\n"
            "Base score: 100 points\n\n"
            "Penalties applied:\n"
            "1. HEART RATE: -20 if avg_heart_rate_bpm > 200 OR avg_heart_rate_bpm < 80\n"
            "2. SLEEP: -15 if estimated_sleep_minutes < 480 (8 hrs) OR > 1100 (18.3 hrs)\n"
            "3. ACTIVITY: -15 if total_daily_steps < 500\n"
            "4. BATTERY: -5 if min_battery_level_pct < 10%\n\n"
            "Final score = 100 - sum of applicable penalties\n"
            "Clamped to range [0, 100]\n\n"
            "Health Status Categories:\n"
            "- Excellent: score >= 85\n"
            "- Good: score >= 70 AND < 85\n"
            "- Needs Attention: score >= 50 AND < 70\n"
            "- Critical Care: score < 50\n\n"
            "Source: dbt_project/models/marts/daily_cat_health.sql (health_scoring CTE)"
        )

    # Fallback for unknown metrics - search docs
    index = get_doc_index()
    context = index.format_context(f"how is {metric_name} calculated", k=3)
    return f"No exact formula found for '{metric_name}'. Related documentation:\n\n{context}"


ALL_TOOLS = [query_gold_layer, search_project_docs, get_cat_directory, explain_metric]

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