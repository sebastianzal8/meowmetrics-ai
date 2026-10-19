"""LLM-as-judge scoring for MeowMetrics RAG evals (local Ollama)."""
from __future__ import annotations

import re

from ai.agent.llm import get_llm
from ai.agent.prompts import LLM_JUDGE_PROMPT

_NUM_RE = re.compile(r"\b(10|[1-9])\b")


def _parse_int_loose(text: str) -> int | None:
    m = _NUM_RE.search(text.strip())
    if m:
        return int(m.group(1))
    return None


def faithfulness_score(question: str, context: str, answer: str) -> float:
    """1-10 score from a small local judge LLM; returns 0.0 on any parsing failure."""
    llm = get_llm()
    prompt = LLM_JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    try:
        response = llm.invoke(prompt)
        value = _parse_int_loose(str(response.content))
        return float(value) if value is not None else 0.0
    except Exception as exc:
        print(f"[judge] scoring failed: {exc}")
        return 0.0
