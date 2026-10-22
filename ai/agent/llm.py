"""Central Ollama client factory (lazy so Streamlit startup stays fast)."""
from __future__ import annotations

from typing import Any

_LLM: Any = None


def get_llm(model: str | None = None, temperature: float | None = None, base_url: str | None = None):
    """Build (or reuse) a ChatOllama instance for the given settings."""
    from langchain_ollama import ChatOllama

    from ai.config import CONFIG

    key_model = model or CONFIG.chat_model
    key_temp = CONFIG.temperature if temperature is None else temperature
    key_base = base_url or CONFIG.ollama_base_url
    global _LLM
    if _LLM is None or getattr(_LLM, "model", None) != key_model:
        _LLM = ChatOllama(
            model=key_model,
            base_url=key_base,
            temperature=key_temp,
            num_ctx=CONFIG.num_ctx,
        )
    return _LLM
