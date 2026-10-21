"""Drift tests: health rules, dbt SQL, mock data and the golden set must agree.

These run without Ollama, Snowflake or network access (DuckDB sandbox only).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai import health_rules as hr
from ai.data_access.mock_data import CATS, build_mock_gold_data
from ai.data_access.sandbox import run_select_in_sandbox
from ai.data_access.sql_guard import sanitize_sql

DBT_SQL = (ROOT / "dbt_project" / "models" / "marts" / "daily_cat_health.sql").read_text(encoding="utf-8")
GOLDEN = yaml.safe_load((ROOT / "ai" / "evals" / "golden_set.yaml").read_text(encoding="utf-8"))
SQL_QUESTIONS = GOLDEN["sql_questions"]


def _ints(pattern: str) -> tuple[int, ...]:
    match = re.search(pattern, DBT_SQL, re.IGNORECASE | re.DOTALL)
    assert match, f"pattern not found in daily_cat_health.sql: {pattern}"
    return tuple(int(g) for g in match.groups())


# ----------------------------------------------------------------------
# health_rules <-> dbt SQL
# ----------------------------------------------------------------------


def test_heart_rate_rule_matches_dbt():
    hr_max, hr_min, penalty = _ints(
        r"avg_heart_rate_bpm\s*>\s*(\d+)\s+OR\s+avg_heart_rate_bpm\s*<\s*(\d+)\s+THEN\s+(\d+)"
    )
    assert (hr.HR_MAX_BPM, hr.HR_MIN_BPM, hr.HR_PENALTY) == (hr_max, hr_min, penalty)


def test_sleep_rule_matches_dbt():
    s_min, s_max, penalty = _ints(
        r"estimated_sleep_minutes\s*<\s*(\d+)\s+OR\s+estimated_sleep_minutes\s*>\s*(\d+)\s+THEN\s+(\d+)"
    )
    assert (hr.SLEEP_MIN_MINUTES, hr.SLEEP_MAX_MINUTES, hr.SLEEP_PENALTY) == (s_min, s_max, penalty)


def test_steps_rule_matches_dbt():
    steps_min, penalty = _ints(r"total_daily_steps\s*<\s*(\d+)\s+THEN\s+(\d+)")
    assert (hr.STEPS_MIN, hr.STEPS_PENALTY) == (steps_min, penalty)


def test_battery_rule_matches_dbt():
    battery_min, penalty = _ints(r"min_battery_level_pct\s*<\s*(\d+)\s+THEN\s+(\d+)")
    assert (hr.BATTERY_MIN_PCT, hr.BATTERY_PENALTY) == (battery_min, penalty)


def test_bands_match_dbt():
    (excellent,) = _ints(r"computed_score\s*>=\s*(\d+)\s+THEN\s+'Excellent'")
    (good,) = _ints(r"computed_score\s*>=\s*(\d+)\s+AND\s+computed_score\s*<\s*\d+\s+THEN\s+'Good'")
    (attention,) = _ints(
        r"computed_score\s*>=\s*(\d+)\s+AND\s+computed_score\s*<\s*\d+\s+THEN\s+'Needs Attention'"
    )
    assert (hr.BAND_EXCELLENT, hr.BAND_GOOD, hr.BAND_ATTENTION) == (excellent, good, attention)


# ----------------------------------------------------------------------
# health_rules behavior
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "args,expected",
    [
        ((135, 800, 3000), 100),
        ((None, 800, 3000), 100),            # NULL never triggers a penalty (SQL semantics)
        ((float("nan"), 800, 3000), 100),
        ((210, 800, 3000), 80),
        ((135, 400, 3000), 85),
        ((135, 1200, 3000), 85),
        ((135, 800, 100), 85),
        ((135, 800, 3000, 5), 95),
        ((210, 400, 100, 5), 45),            # every penalty at once
    ],
)
def test_compute_score(args, expected):
    assert hr.compute_score(*args) == expected


@pytest.mark.parametrize(
    "score,label",
    [(100, "Excellent"), (85, "Excellent"), (84, "Good"), (70, "Good"),
     (69, "Needs Attention"), (50, "Needs Attention"), (49, "Critical Care")],
)
def test_classify_boundaries(score, label):
    assert hr.classify(score) == label


# ----------------------------------------------------------------------
# mock data follows the rules
# ----------------------------------------------------------------------


def test_mock_scores_follow_rules():
    df = build_mock_gold_data()
    for row in df.itertuples():
        expected = hr.compute_score(
            row.avg_heart_rate_bpm, row.estimated_sleep_minutes, row.total_daily_steps
        )
        assert row.daily_feline_health_score == expected
        assert row.health_status_category == hr.classify(expected)


def test_mock_data_exercises_multiple_bands():
    """A demo where every day is 'Excellent' hides bugs; require real variety."""
    statuses = set(build_mock_gold_data()["health_status_category"])
    assert len(statuses) >= 3


def test_mock_data_is_deterministic():
    a = build_mock_gold_data().drop(columns=["daily_summary_id", "reading_date"])
    b = build_mock_gold_data().drop(columns=["daily_summary_id", "reading_date"])
    assert a.equals(b)


# ----------------------------------------------------------------------
# golden set sanity
# ----------------------------------------------------------------------


def test_golden_ids_and_sql_are_unique():
    ids = [q["id"] for q in SQL_QUESTIONS]
    sqls = [" ".join(q["expected_sql"].split()).lower() for q in SQL_QUESTIONS]
    assert len(ids) == len(set(ids)), "duplicate golden question ids"
    assert len(sqls) == len(set(sqls)), "duplicate golden expected_sql"


@pytest.mark.parametrize("item", SQL_QUESTIONS, ids=[q["id"] for q in SQL_QUESTIONS])
def test_golden_sql_returns_rows_for_real_cats(item):
    """An expected query that returns nothing makes the eval pass on empty-vs-empty."""
    known = {c["cat_name"] for c in CATS}
    for name in re.findall(r"cat_name\s*=\s*'([^']+)'", item["expected_sql"]):
        assert name in known, f"{item['id']} references unknown cat {name!r}"
    hardened, _ = sanitize_sql(item["expected_sql"])
    rows = run_select_in_sandbox(hardened)
    assert rows, f"{item['id']} returns 0 rows on the mock data"