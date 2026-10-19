"""Mock data determinism + sandbox execution (no Ollama/no network required)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

from ai.data_access.mock_data import CATS, build_mock_gold_data  # noqa: E402
from ai.data_access.sandbox import run_select_in_sandbox  # noqa: E402
from ai.data_access.sql_guard import sanitize_sql  # noqa: E402


def test_mock_data_is_deterministic_across_processes():
    df1 = build_mock_gold_data()
    df2 = build_mock_gold_data()
    assert len(df1) == len(df2) == len(CATS) * 30
    assert df1["total_daily_steps"].sum() == df2["total_daily_steps"].sum()


def test_expected_shapes_and_ranges():
    df = build_mock_gold_data()
    assert len(df) == len(CATS) * 30
    assert df["daily_feline_health_score"].between(0, 100).all()
    assert df["estimated_sleep_minutes"].ge(0).all()
    assert df["reading_date"].nunique() == 30


def test_sample_sql_executes_in_sandbox():
    valid, _ = sanitize_sql(
        "SELECT cat_name, COUNT(*) AS n FROM BI_TABLEAU_VIEWS "
        "WHERE health_status_category = 'Excellent' GROUP BY cat_name LIMIT 10"
    )
    rows = run_select_in_sandbox(valid)
    assert isinstance(rows, list)
    assert all(len(r) == 2 for r in rows)
