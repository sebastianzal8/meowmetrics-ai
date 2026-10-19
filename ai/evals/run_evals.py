"""MeowMetrics AI eval harness — runs the golden set + logs to MLflow.

Usage (from project root):
    OLLAMA is expected locally; run:  `python -m ai.evals.run_evals`
    Optional flags: --limit N, --skip-judge, --off-mlflow

Scoring logic:
  SQL:     each question -> agent generates SQL -> sql_guard validates -> executed
           against the mock data via DuckDB -> scored by
           (a) result-shape match vs the golden SQL and
           (b) sqlglot AST normalized-match.
  Docs:    retrieval hit rate = expected keyword found in retrieved chunks.
  Judge:   LLM-as-judge faithfulness 1-10 for doc questions (skippable).
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import pandas as pd
import sqlglot
import yaml
from sqlglot import expressions as exp

from ai.agent import agent as agent_module
from ai.agent.tools import search_project_docs
from ai.config import CONFIG
from ai.data_access.mock_data import build_mock_gold_data
from ai.data_access.sandbox import run_select_in_sandbox
from ai.data_access.sql_guard import SQLGuardError, normalize_sql, sanitize_sql
from ai.evals.judge import faithfulness_score

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.yaml"
MLFLOW_TRACKING_DIR = Path(__file__).parent / "mlruns"


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def execute_on_mock_df(sql: str) -> pd.DataFrame:
    valid, _ = sanitize_sql(sql)
    con = get_sandbox_con()
    return con.execute(valid).fetchdf()


def get_sandbox_con():
    """Same engine the agent's tools use (mock catalog MEOW_DB attached)."""
    from ai.data_access.sandbox import get_sandbox
    return get_sandbox()


def _result_signature(df: pd.DataFrame, max_rows: int = 12) -> tuple:
    """Comparable signature: (shape, column multiset, n rows)."""
    cols = tuple(sorted(str(c).lower() for c in df.columns))
    return (df.shape, cols, min(len(df), max_rows))


def score_sql(qitem: dict) -> dict:
    """Runs one SQL golden question via the real agent and scores it."""
    question = qitem["question"]
    expected_df = execute_on_mock_df(qitem["expected_sql"])
    expected_norm = normalize_sql(qitem["expected_sql"])

    result = agent_module.ask(question)
    answer = result.get("answer", "")
    agent_sql = _sql_from_trace(result.get("trace", []))
    if not agent_sql:
        # lenient fallback: only if the answer IS raw SQL (no markdown wrapper)
        if answer.strip().lower().startswith(("select", "with")):
            agent_sql = answer

    metrics = {
        "valid": False, "shape_match": False, "ast_match": False,
        "agent_sql": agent_sql or "", "answer": answer[:400],
        "num_tool_calls": len([t for t in result.get("trace", []) if t.get("type") == "tool_call"]),
    }
    if not agent_sql:
        return metrics

    try:
        valid, effective = sanitize_sql(agent_sql)
    except SQLGuardError as exc:
        metrics["guard_rejects"] = 1
        print(f"[eval {qitem['id']}] guard rejected: {exc}")
        return metrics
    metrics["valid"] = True
    # exec-based score
    try:
        agent_df = execute_on_mock_df(valid)
        metrics["shape_match"] = _result_signature(agent_df) == _result_signature(expected_df)
    except Exception:
        print(f"[eval {qitem['id']}] exec failure:\n{traceback.format_exc(limit=3)}")
    metrics["ast_match"] = normalize_sql(agent_sql) == expected_norm
    return metrics


def _sql_from_trace(trace: list[dict]) -> str | None:
    """Pulls the SQL the agent actually executed (last query_gold_layer call)."""
    sqls: list[str] = []
    for entry in trace:
        if entry.get("type") != "tool_call":
            continue
        for tool_name, args in (entry.get("args") or {}).items():
            if tool_name == "query_gold_layer" and isinstance(args, dict):
                sql = args.get("sql")
                if sql:
                    sqls.append(sql)
    return sqls[-1] if sqls else None


def score_doc(qitem: dict) -> dict:
    question = qitem["question"]
    try:
        context = search_project_docs.func(question)
    except Exception as exc:
        context = f"RETRIEVAL FAILED: {exc}"

    keywords = qitem["expected_doc_keywords"]
    lowered = context.lower()
    hits = [kw for kw in keywords if kw.lower() in lowered]
    retrieval_hit_rate = len(hits) / max(len(keywords), 1)
    return {
        "question": question,
        "retrieval_hit_rate": retrieval_hit_rate,
        "hit_keywords": hits,
        "missed_keywords": [k for k in keywords if k not in hits],
        "context_snippet": context[:800],
    }


def run_evals(limit: int = 0, skip_judge: bool = False, off_mlflow: bool = False) -> dict:
    golden = load_golden_set()
    sql_items = golden.get("sql_questions", [])[:limit or None]
    doc_items = golden.get("doc_questions", [])[:limit or None]

    print(f"Running {len(sql_items)} SQL questions + {len(doc_items)} doc questions ...")

    sql_scores = []
    for item in sql_items:
        s = score_sql(item)
        sql_scores.append({"id": item["id"], **s})
        print(f"  SQL {item['id']}: valid={s['valid']} shape={s['shape_match']} ast={s['ast_match']}")

    doc_scores = []
    for item in doc_items:
        s = score_doc(item)
        if not skip_judge:
            s["faithfulness"] = faithfulness_score(item["question"], s["context_snippet"], agent_module.ask(item["question"])["answer"])
        doc_scores.append({"id": item["id"], "question": item["question"], **s})
        print(f"  DOC {item['id']}: hit_rate={s['retrieval_hit_rate']:.2f}")

    metrics = {
        "sql_valid_rate": _avg(sql_scores, "valid"),
        "sql_shape_match_rate": _avg(sql_scores, "shape_match"),
        "sql_ast_match_rate": _avg(sql_scores, "ast_match"),
        "doc_retrieval_hit_rate": _avg(doc_scores, "retrieval_hit_rate"),
        "doc_faithfulness": _avg(doc_scores, "faithfulness"),
        "n_evaluated": len(sql_scores) + len(doc_scores),
    }
    print("Summary:", metrics)

    if not off_mlflow:
        _log_to_mlflow(metrics, sql_scores, doc_scores)
    return {"metrics": metrics, "sql_scores": sql_scores, "doc_scores": doc_scores}


def _avg(items: list[dict], key: str) -> float:
    vals = [float(i[key]) for i in items if i.get(key) is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _log_to_mlflow(metrics: dict, sql_scores: list, doc_scores: list):
    """Log the run and per-question tables to the project-local MLflow store."""
    import mlflow

    mlflow.set_tracking_uri("sqlite:///ai/evals/mlflow.db")
    mlflow.set_experiment("meowmetrics_ai_evals")
    tags = {"chat_model": CONFIG.chat_model, "embed_model": CONFIG.embed_model}
    with mlflow.start_run(run_name=f"evals_{time.strftime('%Y%m%d_%H%M%S')}", tags=tags):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        if sql_scores:
            mlflow.log_table(pd.DataFrame(sql_scores), artifact_file="sql_scores.json")
        if doc_scores:
            mlflow.log_table(pd.DataFrame(doc_scores), artifact_file="doc_scores.json")
    print(f"MLflow run logged at {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MeowMetrics golden-set evals.")
    parser.add_argument("--limit", type=int, default=0, help="Cap items per section (0 = all).")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM-as-judge (faster).")
    parser.add_argument("--off-mlflow", action="store_true", help="Don't log an MLflow run.")
    args = parser.parse_args()
    run_evals(limit=args.limit, skip_judge=args.skip_judge, off_mlflow=args.off_mlflow)
