"""Validates ai/evals/golden_set.yaml structure (runs in CI without Ollama)."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

from ai.evals.run_evals import GOLDEN_SET_PATH  # noqa: E402


def test_golden_set_wellformed():
    data = yaml.safe_load(GOLDEN_SET_PATH.read_text(encoding="utf-8"))

    sql_items = data.get("sql_questions", [])
    doc_items = data.get("doc_questions", [])
    assert len(sql_items) >= 10, "SQL section should have meaningful coverage"
    assert len(doc_items) >= 5

    for item in sql_items:
        assert item["id"] and item["question"]
        assert isinstance(item["expected_sql"], str) and item["expected_sql"].strip()
        assert item["expected_sql"].strip().lower().startswith("select")

    for item in doc_items:
        assert item["id"] and item["question"]
        assert isinstance(item["expected_doc_keywords"], list) and item["expected_doc_keywords"]


def test_expected_sql_passes_the_guard_itself():
    from ai.data_access.sql_guard import sanitize_sql

    data = yaml.safe_load(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    for item in data.get("sql_questions", []):
        hardened, _ = sanitize_sql(item["expected_sql"])
        assert hardened  # guard accepts the baseline queries
