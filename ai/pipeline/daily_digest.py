"""Daily LLM digest: deterministic health features -> grounded natural-language narrative.

Design principle: the LLM NEVER discovers the data itself. The pipeline it runs
inside computes the health signals (7-day windows over the Gold mart — same
aggregation logic as dbt), the LLM only *narrates* them. That keeps it auditable:
risk flags are non-deterministic-free SQL, the narration is disposable prose.

Outputs `ANALYTICS.LLM_DAILY_SUMMARIES` in Snowflake (prod) or a local CSV in
mock mode, always with provenance columns (model, prompt hash, tokens, ts).

Run directly:   python -m ai.pipeline.daily_digest
Airflow task:   see airflow/dags/meowmetrics_orchestrator.py -> llm_digest group
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from ai.agent.llm import get_llm
from ai.agent.prompts import DIGEST_PROMPT_TEMPLATE
from ai.config import CONFIG
from ai.data_access.mock_data import build_mock_gold_data
from ai.health_rules import (
    BAND_GOOD,
    HR_MAX_BPM,
    HR_MIN_BPM,
    SLEEP_MAX_MINUTES,
    SLEEP_MIN_MINUTES,
    STEPS_MIN,
)


def compute_risk_flags(window: pd.DataFrame) -> list[str]:
    """Rule-based flags — deterministic, SQL-equivalent, no LLM involved.

    Thresholds come from ``ai.health_rules`` (the same values as the dbt score),
    applied to the window averages.
    """
    flags: list[str] = []
    avg_score = window["daily_feline_health_score"].mean()
    avg_steps = window["total_daily_steps"].mean()
    avg_sleep = window["estimated_sleep_minutes"].mean()
    avg_hr = window["avg_heart_rate_bpm"].mean()

    if avg_score < BAND_GOOD:
        flags.append(f"7d_avg_health_score_low:{avg_score:.0f}/100")
    if avg_steps < STEPS_MIN:
        flags.append(f"steps_below_minimum:{avg_steps:.0f}/day (<{STEPS_MIN})")
    if avg_sleep < SLEEP_MIN_MINUTES:
        flags.append(f"undersleeping:{avg_sleep:.0f}min/day (<{SLEEP_MIN_MINUTES})")
    if avg_sleep > SLEEP_MAX_MINUTES:
        flags.append(f"oversleeping:{avg_sleep:.0f}min/day (>{SLEEP_MAX_MINUTES})")
    if avg_hr < HR_MIN_BPM or avg_hr > HR_MAX_BPM:
        flags.append(f"heart_rate_out_of_range:{avg_hr:.0f}bpm")
    if window["health_status_category"].isin(["Needs Attention", "Critical Care"]).mean() >= 0.6:
        flags.append("majority_of_days_negative_status")
    return flags


def _window_rows(bi: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    df = bi.copy()
    df["reading_date"] = pd.to_datetime(df["reading_date"])
    cutoff = df["reading_date"].max() - timedelta(days=lookback_days)
    return df[df["reading_date"] > cutoff]


def generate_digests(lookback_days: int | None = None) -> pd.DataFrame:
    """Build one digest row per cat for the trailing N days."""
    bi = build_mock_gold_data() if not CONFIG.has_snowflake else _load_windows_from_snowflake(lookback_days)
    windows = _window_rows(bi, lookback_days or CONFIG.digest_lookback_days)

    llm = get_llm()
    prompt_hash = hashlib.md5(DIGEST_PROMPT_TEMPLATE.encode()).hexdigest()[:10]
    out: list[dict] = []

    for cat_id, grp in windows.groupby("cat_id", sort=True):
        avg_score = grp["daily_feline_health_score"].mean()
        avg_hr = grp["avg_heart_rate_bpm"].mean()
        avg_steps = grp["total_daily_steps"].mean()
        avg_sleep = grp["estimated_sleep_minutes"].mean()
        flags = compute_risk_flags(grp)
        cat_name = grp["cat_name"].iloc[0]
        breed = grp["breed"].iloc[0]

        prompt = DIGEST_PROMPT_TEMPLATE.format(
            cat_name=cat_name, breed=breed,
            avg_score=round(avg_score, 1),
            trend=flags[0].split(":")[0] if flags else "stable",
            avg_steps=f"{avg_steps:.0f}",
            avg_sleep_minutes=f"{avg_sleep:.0f}",
            avg_hr=f"{avg_hr:.0f}",
            risk_flags=", ".join(flags) if flags else "none",
            steps_min=STEPS_MIN,
            sleep_min=SLEEP_MIN_MINUTES,
            sleep_max=SLEEP_MAX_MINUTES,
            hr_min=HR_MIN_BPM,
            hr_max=HR_MAX_BPM,
        )

        response = llm.invoke(prompt)
        usage = getattr(response, "usage_metadata", None) or {}
        tokens = int(usage.get("total_tokens", 0) or 0)

        _first, _last = grp["daily_feline_health_score"].iloc[0], grp["daily_feline_health_score"].iloc[-1]
        out.append({
            "digest_date": date.today().isoformat(),
            "cat_id": cat_id,
            "cat_name": cat_name,
            "window_start": grp["reading_date"].min().date().isoformat(),
            "window_end": grp["reading_date"].max().date().isoformat(),
            "avg_health_score_7d": round(avg_score, 1),
            "avg_steps_7d": round(avg_steps),
            "avg_sleep_min_7d": round(avg_sleep),
            "avg_heart_rate_7d": round(avg_hr),
            "risk_flags": " | ".join(flags),
            "llm_digest": response.content.strip(),
            "model_name": CONFIG.chat_model,
            "prompt_hash": prompt_hash,
            "tokens_used": tokens,
            "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        })

    return pd.DataFrame(out)


def _load_windows_from_snowflake(lookback_days: int) -> pd.DataFrame:
    """Gold-mart side of the digest (read straight from dbt tables — no BI table needed)."""
    from ai.data_access.gold_client import run_select

    cols, rows = run_select(f"""
        SELECT
            d.cat_id, c.cat_name, c.breed, d.reading_date,
            d.avg_heart_rate_bpm, d.total_daily_steps,
            d.estimated_sleep_minutes, d.daily_feline_health_score,
            d.health_status_category
        FROM MEOW_DB.{CONFIG.sf_schema}.DAILY_CAT_HEALTH AS d
        JOIN MEOW_DB.{CONFIG.sf_schema}.DIM_CATS AS c ON c.cat_id = d.cat_id
        WHERE d.reading_date > CURRENT_DATE - {CONFIG.digest_lookback_days + 1}
    """)
    return pd.DataFrame(rows, columns=cols)


def write_output(df: pd.DataFrame) -> str:
    """Write to Snowflake (overwrite the day's rows via URL params) or local CSV."""
    if CONFIG.has_snowflake:
        import snowflake.connector

        vals = df.where(pd.notnull(df), "None").values.tolist()
        ctx = snowflake.connector.connect(
            user=CONFIG.sf_user, password=CONFIG.sf_password, account=CONFIG.sf_account,
            warehouse=CONFIG.sf_warehouse, database=CONFIG.sf_database,
            schema=CONFIG.sf_schema,
        )
        try:
            cs = ctx.cursor()
            cs.execute("BEGIN")
            cs.execute(f"DELETE FROM MEOW_DB.{CONFIG.sf_schema}.LLM_DAILY_SUMMARIES WHERE digest_date = '{df['digest_date'].iloc[0]}'")
            cs.executemany(
                f"INSERT INTO MEOW_DB.{CONFIG.sf_schema}.LLM_DAILY_SUMMARIES VALUES (" + ",".join(["%s"] * len(df.columns)) + ")",
                [tuple(map(str, r)) for r in vals],
            )
            cs.execute("COMMIT")
        finally:
            cs.close()
            ctx.close()
        return f"snowflake: MEOW_DB.{CONFIG.sf_schema}.LLM_DAILY_SUMMARIES"
    else:
        CONFIG.digest_output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CONFIG.digest_output_csv, index=False)
        return str(CONFIG.digest_output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-cat daily LLM digests.")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of cats (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of writing.")
    args = parser.parse_args()

    digests = generate_digests()
    if args.limit and args.limit < len(digests):
        digests = digests.head(args.limit)
    if args.dry_run:
        import json

        print(json.dumps(digests.to_dict(orient="records"), indent=2, default=str))
    else:
        print(f"Wrote {len(digests)} digest rows -> {write_output(digests)}")