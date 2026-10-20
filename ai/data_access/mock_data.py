"""Mock Gold-layer generator — the single source of truth for demo data.

Extracted from ``dashboard/app.py`` so the dashboard, the AI copilot and the
eval harness all see exactly the same data when Snowflake is offline.

One deliberate improvement over the original: the seed is derived from an
MD5 of ``cat_id`` instead of Python's ``hash()`` (which changes across
interpreter runs), so visualizations and evals are deterministic.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CATS = [
    {"cat_id": "CAT001", "cat_name": "Gaia", "breed": "Chanda", "gender": "female", "age_years": 3, "human_equivalent_age": 28, "weight_lbs": 8.4, "weight_category": "healthy weight"},
    {"cat_id": "CAT002", "cat_name": "Michin", "breed": "Stray", "gender": "male", "age_years": 5, "human_equivalent_age": 36, "weight_lbs": 16.2, "weight_category": "overweight"},
    {"cat_id": "CAT003", "cat_name": "Milo", "breed": "Persian", "gender": "male", "age_years": 8, "human_equivalent_age": 48, "weight_lbs": 12.8, "weight_category": "overweight"},
    {"cat_id": "CAT004", "cat_name": "Bella", "breed": "Ragdoll", "gender": "female", "age_years": 1, "human_equivalent_age": 15, "weight_lbs": 9.1, "weight_category": "healthy weight"},
    {"cat_id": "CAT005", "cat_name": "Simba", "breed": "Tabby", "gender": "male", "age_years": 12, "human_equivalent_age": 64, "weight_lbs": 17.5, "weight_category": "obese"},
]

DAYS_OF_TELEMETRY = 30


def _stable_seed(cat_id: str) -> int:
    """Deterministic per-cat seed (stable across Python processes)."""
    return int(hashlib.md5(cat_id.encode()).hexdigest()[:8], 16) % 1000


def build_mock_gold_data() -> pd.DataFrame:
    """Generates a mock ``BI_TABLEAU_VIEWS``-shaped dataset."""
    records = []
    start_date = datetime.now() - timedelta(days=DAYS_OF_TELEMETRY)

    for cat in CATS:
        np.random.seed(_stable_seed(cat["cat_id"]))
        base_steps = 4000 if cat["cat_id"] in ["CAT001", "CAT004"] else 1500  # Siamese/Ragdoll active, Persian lazy
        base_sleep = 750 if cat["cat_id"] in ["CAT001", "CAT004"] else 950
        base_hr = 135 if cat["cat_id"] != "CAT005" else 155

        for d in range(DAYS_OF_TELEMETRY):
            current_day = start_date + timedelta(days=d)
            steps = int(np.random.normal(base_steps, 500))
            sleep_min = int(np.random.normal(base_sleep, 60))
            active_min = int(np.random.normal(1440 - sleep_min - 200, 30))

            steps = max(0, steps)
            sleep_min = min(1440, max(0, sleep_min))
            active_min = min(1440 - sleep_min, max(0, active_min))
            lazy_min = 1440 - sleep_min - active_min

            hr = int(np.random.normal(base_hr, 10))
            purr = round(max(0.0, float(np.random.normal(25.0, 5.0))), 2)

            # Mirror of the dbt health scoring logic for consistency of the demo
            score = 100
            if hr > 150 or hr < 110:
                score -= 15
            if sleep_min < 480 or sleep_min > 1100:
                score -= 15
            if steps < 1000:
                score -= 15
            score = max(0, min(100, score))

            if score >= 85:
                status = "Excellent"
            elif score >= 70:
                status = "Good"
            elif score >= 50:
                status = "Needs Attention"
            else:
                status = "Critical Care"

            records.append({
                "daily_summary_id": f"{cat['cat_id']}_{current_day.strftime('%Y%m%d')}",
                "cat_id": cat["cat_id"],
                "cat_name": cat["cat_name"],
                "breed": cat["breed"],
                "gender": cat["gender"],
                "age_years": cat["age_years"],
                "human_equivalent_age": cat["human_equivalent_age"],
                "weight_lbs": cat["weight_lbs"],
                "weight_category": cat["weight_category"],
                "reading_date": current_day.date(),
                "avg_heart_rate_bpm": hr,
                "total_daily_steps": steps,
                "avg_purr_frequency_hz": purr,
                "estimated_sleep_minutes": sleep_min,
                "estimated_active_minutes": active_min,
                "estimated_lazy_minutes": lazy_min,
                "daily_feline_health_score": score,
                "health_status_category": status,
                "sleep_percentage_of_day": round((sleep_min / 1440) * 100, 1),
                "active_percentage_of_day": round((active_min / 1440) * 100, 1),
            })

    return pd.DataFrame(records)
