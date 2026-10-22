"""Guardrails for agent-generated SQL before it ever touches the warehouse.

Strategy is defense-in-depth:
1. regex pre-filter for forbidden whole-word statements (INSERT/DROP/...),
2. strict grammar check via sqlglot: exactly one statement, rooted at SELECT,
3. LIMIT injection + row cap.

The guard never *fixes* queries silently — anything suspicious is rejected so
the agent can retry with feedback.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import expressions as exp

# Whole words only — avoids matching e.g. a column named "created_at"
_FORBIDDEN = re.compile(
    r"""\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke|
        copy|call|put|get|set|unset|use|show|describe|explain|execute|begin|
        commit|rollback|into)\b""",
    re.IGNORECASE | re.VERBOSE,
)

FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)

DEFAULT_MAX_LIMIT = 200


class SQLGuardError(ValueError):
    """Raised when a generated query fails validation."""


def sanitize_sql(raw_sql: str, max_rows: int = DEFAULT_MAX_LIMIT) -> tuple[str, int]:
    """Validate and harden a single SELECT statement. Returns (sql, limit)."""
    if not raw_sql or not raw_sql.strip():
        raise SQLGuardError("Empty query.")

    sql = FENCE_RE.sub("", raw_sql.strip()).strip()
    sql = sql.strip("` \n\r\t")

    # Fix various special characters that LLMs may generate
    sql = (sql
        .replace('\u2018', "'")   # left single quotation mark
        .replace('\u2019', "'")   # right single quotation mark
        .replace('\u201c', '"')   # left double quotation mark
        .replace('\u201d', '"')   # right double quotation mark
        .replace('\u2013', '-')   # en dash
        .replace('\u2014', '-')   # em dash
        .replace('\u2026', '...') # horizontal ellipsis
    )

    # Whole-word pre-filter — cheap tripwire before the grammar check.
    if _FORBIDDEN.search(sql):
        raise SQLGuardError("Query contains forbidden write/admin keywords.")

    # Hand the raw text to sqlglot: its tokenizer understands string literals,
    # so semicolons/quotes INSIDE them are safe. parse() returns a statement
    # list, which also gives us an exact multi-statement check.
    try:
        statements = sqlglot.parse(sql, dialect="snowflake")
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:  # sqlglot raises ParseError/TokenError etc.
        raise SQLGuardError(f"SQL parse failed: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) == 0:
        raise SQLGuardError("SQL parse produced nothing.")
    if len(statements) > 1:
        raise SQLGuardError(f"Multiple statements are not allowed ({len(statements)} found).")

    parsed = statements[0]

    # sqlglot returns a Select for both bare SELECTs and WITH ... SELECT
    if not isinstance(parsed, exp.Select):
        raise SQLGuardError(f"Only SELECT queries are allowed, got {type(parsed).__name__}.")

    # Enforce a LIMIT and cap it (Limit.expression is a numeric literal)
    limit = parsed.args.get("limit")
    if limit is not None:
        try:
            requested = int(limit.expression.this)
        except (AttributeError, TypeError, ValueError):
            requested = -1
        effective = max(min(requested, max_rows), 1)
        limit.set("expression", exp.Literal.number(effective))
        root = parsed
    else:
        effective = max_rows
        root = parsed.limit(max_rows)

    return root.sql(dialect="snowflake"), effective


def normalize_sql(sql: str) -> str:
    """Canonical form used by eval comparisons (lowercase keys, fresh dialect)."""
    valid, _ = sanitize_sql(sql)
    return sqlglot.parse_one(valid, dialect="snowflake").sql(dialect="snowflake", normalize=True).lower()
