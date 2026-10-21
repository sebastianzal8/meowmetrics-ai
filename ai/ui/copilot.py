"""Streamlit chat UI for the MeowMetrics copilot agent."""
from __future__ import annotations

import json

import streamlit as st


def _css():
    st.markdown(
        """<style>
        .copilot-note {font-size: 0.85em; opacity: 0.7;}
        .meow-source {font-size: 0.78em; opacity: 0.65; font-family: monospace;}
        </style>""",
        unsafe_allow_html=True,
    )


def render_copilot() -> None:
    _css()
    st.subheader("🤖 Ask the Pipeline (Agentic RAG Copilot)")
    st.caption(
        "Local Ollama agent that can execute read-only SQL over the Gold layer, "
        "search the repo's own engineering docs, and cite sources."
    )

    col_model, col_reset = st.columns([3, 1])
    with col_model:
        model_choice = st.selectbox(
            "Model",
            ["llama3.2:3b", "llama3.1:8b", "qwen2.5:7b", "qwen2.5-coder:7b"],
            index=0,
            help="Larger local models generate better SQL but run slower on CPU.",
        )
    with col_reset:
        st.write("")
        if st.button("🧹 New chat"):
            st.session_state.copilot_messages = []
            st.session_state.copilot_trace = []
            st.session_state["copilot_model"] = None
            st.rerun()

    # Reload agent when the model changes
    if st.session_state.get("copilot_model") != model_choice:
        st.session_state["copilot_model"] = model_choice
    from ai.agent import agent as agent_module
    from ai.agent.llm import get_llm
    get_llm(model=model_choice)  # warm the/cache the selected model
    agent_module._AGENT = None  # force rebuild for the chosen model

    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = []
    if "copilot_trace" not in st.session_state:
        st.session_state.copilot_trace = []

    # Render conversation history
    for i, turn in enumerate(st.session_state.copilot_messages):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                st.markdown(
                    "**Sources:** " +
                    " · ".join(f"`{s}`" for s in turn["sources"])
                )

    question = st.chat_input("Ask about your cat telemetry or the pipeline...")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                ask_fn = agent_module.ask
                result = ask_fn(question)
                placeholder.markdown(result["answer"] or "_(no answer)_")
                if result["sources"]:
                    st.markdown(
                        "**Sources:** " +
                        " · ".join(f"`{s}`" for s in result["sources"])
                    )
            except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                result = {"answer": "", "trace": [], "sources": []}
                placeholder.error(
                    f"Agent error: {exc}\n\nCheck that Ollama is running "
                    f"(`ollama serve`, port 11434) and the model is pulled."
                )

    st.session_state.copilot_messages.append({
        "role": "user", "content": question, "sources": [],
    })
    st.session_state.copilot_messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "sources": result.get("sources", []),
    })
    trace = result.get("trace", [])
    if trace:
        st.session_state.copilot_trace.append({"q": question, "trace": trace})
        with st.expander("🛠 Agent trace (tools, args, observations)"):
            st.json(json.dumps(trace, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    import pathlib as _p
    import sys as _sys
    _sys.path.insert(0, str(_p.Path(__file__).resolve().parents[2]))
    render_copilot()
