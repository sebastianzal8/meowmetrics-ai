"""Explicit LangGraph StateGraph ReAct Agent with Langfuse tracing.

This replaces the prebuilt `create_react_agent` with an explicit graph:
- Model node: calls the LLM with tools bound
- Tool node: executes tool calls
- Conditional edge: routes back to model or ends
- Langfuse tracing for observability
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from ai.agent.llm import get_llm
from ai.agent.models import AgentResponse, AgentStep, ToolCall, ToolResult
from ai.agent.prompts import SYSTEM_PROMPT
from ai.agent.tools import ALL_TOOLS, extract_sources

# Try to import Langfuse; make it optional
try:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    LangfuseCallbackHandler = None


class AgentState(BaseModel):
    """State for the ReAct agent graph."""
    messages: list[BaseMessage] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    sql_used: list[str] = Field(default_factory=list)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    final_response: dict | None = Field(default=None)

    class Config:
        arbitrary_types_allowed = True


def _get_langfuse_handler() -> Any | None:
    """Initialize Langfuse callback handler if configured."""
    if not LANGFUSE_AVAILABLE:
        return None
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if not public_key or not secret_key:
        return None
    
    try:
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        return LangfuseCallbackHandler(langfuse=langfuse)
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return None


def build_agent_graph() -> StateGraph:
    """Build the explicit ReAct agent StateGraph."""
    
    # Get the LLM with tools bound
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    # Tool node for executing tool calls
    tool_node = ToolNode(ALL_TOOLS)
    
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Decide whether to call tools or finish."""
        last_message = state.messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"
    
    def call_model(state: AgentState, config: RunnableConfig) -> dict:
        """Invoke the LLM with the current conversation state."""
        # Add system prompt if this is the first call
        messages = state.messages
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        # Get Langfuse callback if available
        langfuse_handler = _get_langfuse_handler()
        callbacks = config.get("callbacks", [])
        if langfuse_handler:
            callbacks = callbacks + [langfuse_handler]
        
        # Invoke model
        response = llm_with_tools.invoke(
            messages,
            config=RunnableConfig(callbacks=callbacks, metadata=config.get("metadata", {}))
        )
        
        return {"messages": [response]}
    
    def call_tools(state: AgentState) -> dict:
        """Execute tool calls and record steps."""
        last_message = state.messages[-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}
        
        # Record tool calls in steps
        new_steps = []
        for tc in last_message.tool_calls:
            step = AgentStep(
                type="tool_call",
                content=f"Calling {tc['name']}",
                tool_calls=[ToolCall(name=tc['name'], args=tc['args'], id=tc['id'])]
            )
            new_steps.append(step)
            
            # Track SQL used
            if tc['name'] == 'query_gold_layer' and 'sql' in tc['args']:
                state.sql_used.append(tc['args']['sql'])
        
        # Execute tools
        tool_messages = tool_node.invoke({"messages": state.messages})
        
        # Record tool results
        for tm in tool_messages.get("messages", []):
            if isinstance(tm, ToolMessage):
                step = AgentStep(
                    type="tool_result",
                    content=str(tm.content)[:1200],
                    tool_result=ToolResult(
                        tool=tm.name or "unknown",
                        content=str(tm.content)[:1200],
                        tool_call_id=tm.tool_call_id
                    )
                )
                new_steps.append(step)
        
        return {
            "messages": tool_messages.get("messages", []),
            "steps": state.steps + new_steps
        }
    
    def format_final_response(state: AgentState) -> dict:
        """Format the final response as a Pydantic object."""
        # Check if explain_metric was called - if so, use its result as the primary answer
        explain_metric_result = None
        for step in state.steps:
            if step.type == "tool_call" and step.tool_calls:
                for tc in step.tool_calls:
                    if tc.name == "explain_metric":
                        # Find the corresponding tool result
                        for result_step in state.steps:
                            if (result_step.type == "tool_result" and 
                                result_step.tool_result and 
                                result_step.tool_result.tool_call_id == tc.id):
                                explain_metric_result = result_step.tool_result.content
                                break
                        break
                if explain_metric_result:
                    break
        
        # Extract answer from the last AI message without tool calls
        answer = ""
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                content = msg.content
                if isinstance(content, str):
                    answer = content
                elif isinstance(content, list):
                    answer = "\n".join(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    )
                break
        
        # If explain_metric was called but the model's answer doesn't include its content,
        # use the explain_metric result as the primary answer
        if explain_metric_result and explain_metric_result not in answer:
            # Prepend the explain_metric result to the answer
            if answer:
                answer = explain_metric_result + "\n\n" + answer
            else:
                answer = explain_metric_result
        
        # Extract sources from tool results
        sources = extract_sources([
            step.tool_result.content for step in state.steps 
            if step.type == "tool_result" and step.tool_result
        ])
        
        # Build response
        response = AgentResponse(
            answer=answer or "No answer generated.",
            sql_used=state.sql_used,
            sources=sources,
            trace=state.steps,
            metadata={
                "model": getattr(get_llm(), "model", "unknown"),
                "trace_id": state.trace_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        
        return {"final_response": response.model_dump()}
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("model", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_node("format_response", format_final_response)
    
    # Add edges
    workflow.set_entry_point("model")
    workflow.add_conditional_edges(
        "model",
        should_continue,
        {
            "tools": "tools",
            "end": "format_response",
        }
    )
    workflow.add_edge("tools", "model")
    workflow.add_edge("format_response", END)
    
    return workflow.compile()


# Singleton for the compiled graph
_GRAPH = None


def get_agent_graph():
    """Get or create the compiled agent graph."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agent_graph()
    return _GRAPH


def ask(question: str, recursion_limit: int = 25) -> dict:
    """Invoke the agent and return a structured response.
    
    Args:
        question: The user's question
        recursion_limit: Maximum number of agent steps
        
    Returns:
        Dictionary with answer, sql_used, sources, trace, and metadata
    """
    from ai.agent.tools import _reset_failures
    
    _reset_failures()
    graph = get_agent_graph()
    
    # Initial state
    initial_state = AgentState(
        messages=[HumanMessage(content=question)],
        trace_id=str(uuid.uuid4())
    )
    
    # Invoke the graph
    config = RunnableConfig(
        recursion_limit=recursion_limit,
        metadata={"question": question}
    )
    
    result = graph.invoke(initial_state, config=config)
    
    return result.get("final_response", {
        "answer": "Error: No response generated",
        "sql_used": [],
        "sources": [],
        "trace": [],
        "metadata": {}
    })