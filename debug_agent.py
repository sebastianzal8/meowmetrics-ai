"""Debug script for the agent loop (temporary)."""
import sys

sys.path.insert(0, ".")

from langchain_core.messages import AIMessage

from ai.agent.llm import get_llm
from ai.agent.tools import ALL_TOOLS
from ai.agent.prompts import SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, ToolMessage

llm = get_llm()
tools_by_name = {t.name: t for t in ALL_TOOLS}

# Step 1: does the model emit tool_calls at all?
msgs = [
    SystemMessage(content=SYSTEM_PROMPT),
    ("user", "What is the average daily sleep duration for each cat, in minutes?"),
]
resp = llm.bind_tools(ALL_TOOLS).invoke(msgs)
print("STEP1 tool_calls:", getattr(resp, "tool_calls", None))
print("STEP1 content:", str(resp.content)[:400])

# Step 2: if it called a tool, actually execute ONE step manually
if getattr(resp, "tool_calls", None):
    tc = resp.tool_calls[0]
    result = tools_by_name[tc["name"]].invoke(tc["args"])
    print("STEP2 tool result head:", str(result)[:500])
    msgs2 = msgs + [resp, ToolMessage(content=result, tool_call_id=tc["id"])]
    resp2 = llm.bind_tools(ALL_TOOLS).invoke(msgs2)
    print("STEP2 round2 tool_calls:", getattr(resp2, "tool_calls", None))
    print("STEP2 round2 content:", str(resp2.content)[:600])
