"""The MeowMetrics copilot: a LangGraph ReAct agent over Ollama.

Exposes two entry points used by the dashboard and the eval harness:
  - get_agent(): lazily-built agent (rebuilt when the model changes)
  - ask(question): returns {"answer", "trace", "sources"}
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from ai.agent.llm import get_llm
from ai.agent.prompts import SYSTEM_PROMPT
from ai.agent.tools import ALL_TOOLS, extract_sources

_AGENT: Any = None


def get_agent():
    """Lazily create the ReAct agent; rebuild if the model changed."""
    global _AGENT
    from langgraph.prebuilt import create_react_agent

    if _AGENT is None:
        llm = get_llm()
        _AGENT = create_react_agent(
            model=llm,
            tools=ALL_TOOLS,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
        )
    return _AGENT


def _block_text(block: Any) -> str:
    """Ollama sometimes returns content as a list of typed blocks."""
    if isinstance(block, str):
        return block
    if isinstance(block, list):
        return "\n".join(b.get("text", "") for b in block if isinstance(b, dict))
    return str(block or "")


def ask(question: str, recursion_limit: int = 25) -> dict:
    """Invoke the agent and return a normalized answer + execution trace."""
    from ai.agent.tools import _reset_failures

    _reset_failures()  # per-question failure ledger
    agent = get_agent()
    state = agent.invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": recursion_limit},
    )

    trace: list[dict] = []
    tool_call_names: dict[str, str] = {}
    answer = ""

    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                step = {"type": "tool_call", "tools": [tc["name"] for tc in tool_calls]}
                for tc in tool_calls:
                    tool_call_names[tc["id"]] = tc["name"]
                    step.setdefault("args", {})[tc["name"]] = tc.get("args", {})
                trace.append(step)
            text = _block_text(msg.content)
            if text.strip():
                if tool_calls:  # interim reasoning before acting
                    trace.append({"type": "thought", "text": text.strip()})
                else:
                    answer = text.strip()
        elif isinstance(msg, ToolMessage):
            trace.append({
                "type": "tool_result",
                "tool": tool_call_names.get(msg.tool_call_id, msg.name or "tool"),
                "content": _block_text(msg.content)[:1200],
            })

    sources = extract_sources([
        t.get("content", "") for t in trace if t.get("type") == "tool_result"
    ])

    return {"answer": answer, "trace": trace, "sources": sources}
