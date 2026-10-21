"""The MeowMetrics copilot: a LangGraph ReAct agent over Ollama (explicit StateGraph).

Exposes two entry points used by the dashboard and the eval harness:
  - get_agent(): lazily-built agent (rebuilt when the model changes)
  - ask(question): returns structured AgentResponse with answer, sql_used, sources, trace, metadata
"""
from __future__ import annotations

from typing import Any

from ai.agent.graph import ask as _graph_ask
from ai.agent.graph import get_agent_graph
from ai.agent.models import AgentResponse

# Backward compatibility: the old global agent reference
_AGENT: Any = None


def get_agent():
    """Lazily create the ReAct agent; rebuild if the model changed.
    
    Now returns the compiled StateGraph instead of prebuilt create_react_agent.
    """
    global _AGENT
    if _AGENT is None:
        _AGENT = get_agent_graph()
    return _AGENT


def ask(question: str, recursion_limit: int = 25) -> dict:
    """Invoke the agent and return a normalized answer + execution trace.
    
    Returns a structured dict matching AgentResponse model:
    {
        "answer": str,
        "sql_used": list[str],
        "sources": list[str],
        "trace": list[dict],
        "metadata": dict
    }
    """
    return _graph_ask(question, recursion_limit=recursion_limit)


# Re-export for backward compatibility
from ai.agent.models import AgentStep, ToolCall, ToolResult

__all__ = ["AgentResponse", "AgentStep", "ToolCall", "ToolResult", "ask", "get_agent"]