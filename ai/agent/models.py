"""Pydantic response models for the agent."""
from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation."""
    name: str
    args: dict[str, Any]
    id: str | None = None


class ToolResult(BaseModel):
    """Result of a tool execution."""
    tool: str
    content: str
    tool_call_id: str | None = None


class AgentStep(BaseModel):
    """One step in the agent's reasoning trace."""
    type: Literal["tool_call", "tool_result", "thought", "final"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None


class AgentResponse(BaseModel):
    """Structured response from the agent."""
    answer: str
    sql_used: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    trace: list[AgentStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra: ClassVar[dict] = {
            "example": {
                "answer": "Gaia had the most steps at 33,571.",
                "sql_used": [
                    "SELECT cat_name, SUM(total_daily_steps) AS total_steps FROM BI_TABLEAU_VIEWS WHERE reading_date >= '2026-10-13' GROUP BY cat_name ORDER BY total_steps DESC LIMIT 5"
                ],
                "sources": ["BI_TABLEAU_VIEWS"],
                "trace": [
                    {"type": "tool_call", "content": "query_gold_layer", "tool_calls": [{"name": "query_gold_layer", "args": {"sql": "..."}, "id": "call_1"}]},
                    {"type": "tool_result", "content": "Gaia | 33571", "tool_result": {"tool": "query_gold_layer", "content": "Gaia | 33571", "tool_call_id": "call_1"}}
                ],
                "metadata": {"model": "llama3.2:3b", "recursion_limit": 25}
            }
        }