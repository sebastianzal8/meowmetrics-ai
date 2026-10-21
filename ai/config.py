"""Central configuration for the MeowMetrics AI layer.

All knobs are environment-driven so the same code runs in three modes:
- full pipeline (Snowflake + Airflow),
- local demo (mock Gold layer), and
- CI (no LLM, no network).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


@dataclass
class Config:
    # --- LLM runtime (local Ollama) ---
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = os.getenv("AI_CHAT_MODEL", "llama3.2:3b")
    embed_model: str = os.getenv("AI_EMBED_MODEL", "nomic-embed-text")
    temperature: float = float(os.getenv("AI_TEMPERATURE", "0"))

    # --- Snowflake (read-only consumer; empty envs => mock mode) ---
    sf_user: str = os.getenv("SF_USER", "")
    sf_password: str = os.getenv("SF_PASSWORD", "")
    sf_account: str = os.getenv("SF_ACCOUNT", "")
    sf_warehouse: str = os.getenv("SF_WAREHOUSE", "MEOW_WH")
    sf_database: str = os.getenv("SF_DATABASE", "MEOW_DB")
    sf_schema: str = os.getenv("SF_SCHEMA", "ANALYTICS")

    # Sandbox guardrails for agent-generated SQL
    max_rows_per_query: int = int(os.getenv("AI_MAX_ROWS", "200"))
    sql_timeout_seconds: float = float(os.getenv("AI_SQL_TIMEOUT", "10"))

    # --- RAG index (local Chroma) ---
    index_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "ai" / "index_cache"
    )
    retrieval_k: int = int(os.getenv("AI_RETRIEVAL_K", "4"))

    # --- Daily digest pipeline ---
    digest_output_csv: Path = field(
        default_factory=lambda: PROJECT_ROOT / "ai" / "pipeline" / "output" / "llm_daily_summaries_mock.csv"
    )
    digest_lookback_days: int = int(os.getenv("AI_DIGEST_LOOKBACK", "7"))

    # --- Evals ---
    timebox_eval_questions: int = 0  # 0 = all

    @property
    def has_snowflake(self) -> bool:
        """True when real warehouse credentials are present."""
        return bool(self.sf_user and sf_creds_complete())

    @classmethod
    def load(cls) -> Config:
        return cls()


def sf_creds_complete(config: Config | None = None) -> bool:
    """Snowflake is only in play when user+password+account are all provided."""
    if config is not None:
        return bool(config.sf_user and config.sf_password and config.sf_account)
    return all(os.getenv(k) for k in ("SF_USER", "SF_PASSWORD", "SF_ACCOUNT"))


CONFIG = Config.load()
