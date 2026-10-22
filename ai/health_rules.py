"""Single source of truth for the feline health-score rules.

These constants mirror ``dbt_project/models/marts/daily_cat_health.sql`` (the
production definition). The mock data generator, the daily digest flags and the
LLM prompts all import from here, so the numbers a user sees, the numbers the
copilot explains and the numbers the digest narrates cannot silently diverge.

``tests/test_health_rules.py`` parses the dbt SQL and fails CI if these
constants drift from it.

Note: the dbt score also subtracts ``BATTERY_PENALTY`` when the collar battery
is low. ``BI_TABLEAU_VIEWS`` does not expose battery, so the mock data cannot
apply that term (``min_battery_pct`` is optional in :func:`compute_score`).
"""
from __future__ import annotations

import math

# --- Penalty rules (same values as daily_cat_health.sql) ---------------------
HR_MIN_BPM = 80
HR_MAX_BPM = 200
HR_PENALTY = 20

SLEEP_MIN_MINUTES = 480
SLEEP_MAX_MINUTES = 1100
SLEEP_PENALTY = 15

STEPS_MIN = 500
STEPS_PENALTY = 15

BATTERY_MIN_PCT = 10
BATTERY_PENALTY = 5

# --- Status bands (lower bounds, inclusive) -----------------------------------
BAND_EXCELLENT = 85
BAND_GOOD = 70
BAND_ATTENTION = 50


def _missing(value: float | None) -> bool:
    """SQL semantics: a NULL never satisfies a comparison, so it earns no penalty."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def compute_score(
    avg_hr: float | None,
    sleep_minutes: float | None,
    steps: float | None,
    min_battery_pct: float | None = None,
) -> int:
    """Python twin of the dbt scoring CASE expressions, clamped to 0-100."""
    score = 100
    if not _missing(avg_hr) and (avg_hr > HR_MAX_BPM or avg_hr < HR_MIN_BPM):
        score -= HR_PENALTY
    if not _missing(sleep_minutes) and (
        sleep_minutes < SLEEP_MIN_MINUTES or sleep_minutes > SLEEP_MAX_MINUTES
    ):
        score -= SLEEP_PENALTY
    if not _missing(steps) and steps < STEPS_MIN:
        score -= STEPS_PENALTY
    if not _missing(min_battery_pct) and min_battery_pct < BATTERY_MIN_PCT:
        score -= BATTERY_PENALTY
    return max(0, min(100, score))


def classify(score: float) -> str:
    """Map a 0-100 score to its status band (same bands as the dbt model)."""
    if score >= BAND_EXCELLENT:
        return "Excellent"
    if score >= BAND_GOOD:
        return "Good"
    if score >= BAND_ATTENTION:
        return "Needs Attention"
    return "Critical Care"


def describe_rules() -> str:
    """Plain-language rules text injected into prompts (never hand-copied)."""
    return (
        "The daily score starts at 100 and subtracts penalties: "
        f"{HR_PENALTY} if average heart rate is outside {HR_MIN_BPM}-{HR_MAX_BPM} bpm, "
        f"{SLEEP_PENALTY} if sleep is outside {SLEEP_MIN_MINUTES}-{SLEEP_MAX_MINUTES} minutes, "
        f"{STEPS_PENALTY} if daily steps are below {STEPS_MIN}, "
        f"{BATTERY_PENALTY} if the collar battery is below {BATTERY_MIN_PCT}%. "
        f"Bands: {BAND_EXCELLENT}+ Excellent, {BAND_GOOD}-{BAND_EXCELLENT - 1} Good, "
        f"{BAND_ATTENTION}-{BAND_GOOD - 1} Needs Attention, below {BAND_ATTENTION} Critical Care."
    )