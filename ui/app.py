import asyncio
import os

import streamlit as st

from agent.types import AgentRun, Step
from observability.replay import build_replay_gif
from scripts.run_task import run_task


def format_step_line(step: Step) -> str:
    status = "OK" if step.action_result is None or step.action_result.success else "FAIL"
    return f"[Step {step.number}] {step.action_type}: {step.action_input} -> {status}"


def run_task_sync(task: str, headless: bool, max_steps: int | None) -> AgentRun:
    return asyncio.run(run_task(task, headless=headless, max_steps=max_steps))


def _render_page() -> None:
    st.set_page_config(page_title="Autonomous Web Agent", layout="wide")
    st.title("Autonomous Web Agent")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Run Settings")
        headless = st.checkbox("Headless", value=True)
        max_steps = st.number_input("Max steps", min_value=1, max_value=25, value=25)
        st.caption("Live benchmark results are pending a funded Gemini key.")

    task_text = st.text_area(
        "Task",
        placeholder="Go to https://example.com and report the page title",
        height=120,
    )

    if st.button("Run", type="primary") and task_text.strip():
        with st.spinner("Agent working..."):
            run = run_task_sync(task_text, headless=headless, max_steps=int(max_steps))
        st.session_state.history.insert(0, run)

    if st.session_state.history:
        latest = st.session_state.history[0]
        _render_run(latest)
        _render_history(st.session_state.history)


def _render_run(run: AgentRun) -> None:
    st.subheader("Result")
    cols = st.columns(4)
    cols[0].metric("Success", str(run.success))
    cols[1].metric("Steps", run.total_actions)
    cols[2].metric("Tokens", run.total_tokens)
    cols[3].metric("Duration", f"{run.duration_seconds:.1f}s")
    st.write(run.result or "(no final result)")
    if run.final_url:
        st.caption(f"Final URL: {run.final_url}")

    st.subheader("Step Trace")
    for step in run.steps:
        with st.expander(format_step_line(step)):
            st.write(step.reasoning_text or "(no reasoning text)")
            if step.action_result and step.action_result.error:
                st.code(step.action_result.error)
            if step.action_result and step.action_result.screenshot_b64:
                st.image(f"data:image/png;base64,{step.action_result.screenshot_b64}")

    with st.expander("Raw steps"):
        st.json([step.__dict__ for step in run.steps])

    trace_dir = os.path.join("results", "traces", _safe_session_id(run.task))
    if os.path.exists(os.path.join(trace_dir, "trace.jsonl")):
        if st.button("Build replay GIF"):
            out_path = os.path.join(trace_dir, "replay.gif")
            st.success(build_replay_gif(trace_dir, out_path))
            st.image(out_path)


def _render_history(history: list[AgentRun]) -> None:
    st.subheader("History")
    for index, run in enumerate(history[:10], start=1):
        status = "OK" if run.success else "FAIL"
        st.write(f"{index}. {status} - {run.task[:100]}")


def _safe_session_id(task: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in task.lower())
    return cleaned.strip("_")[:80] or "latest"


if __name__ == "__main__":
    _render_page()
