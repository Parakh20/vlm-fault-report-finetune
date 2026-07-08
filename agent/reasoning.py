import hashlib
import time

from agent.actions import ACTION_SCHEMAS, _TEXT_RETURNING, dispatch_action
from agent.browser import BrowserSession
from agent.config import Settings, build_router
from agent.llm.base import LLMResponse
from agent.llm.router import LLMRouter
from agent.memory import TaskMemory
from agent.planner import PlannerAgent
from agent.types import ActionResult, AgentRun, Step
from agent.validation import validate_action
from agent.verifier import VerifierAgent

MAX_AGENT_STEPS = 25

SYSTEM_PROMPT = """You are an autonomous web agent. You control a real browser
to complete tasks. At each step you see:
- The current URL and page title
- A list of interactive elements on the page (with IDs)
- The accessibility tree of the page
- A screenshot with elements annotated

Your job: decide the single best next action to make progress
toward the task. Think step by step before acting:
1. Where am I? What does this page show?
2. What progress have I made so far?
3. What is the next logical step toward the goal?
4. Which action achieves that step?

Rules:
- One action per turn, no batching
- If lost: go_back or search_web to reorient
- If a page is slow: use wait() before interacting
- If task requires information: use get_page_text() to read it
- When task is fully complete: call task_complete with result
- If task is impossible (login required, CAPTCHA, etc.): task_failed
- Never loop on the same action more than 3 times"""


def _elements_as_text(elements) -> str:
    lines = []
    for element in elements:
        bits = [f'id="{element.id}"', f'tag="{element.tag}"']
        if element.text:
            bits.append(f'text="{element.text[:60]}"')
        if element.placeholder:
            bits.append(f'placeholder="{element.placeholder}"')
        if element.href:
            bits.append(f'href="{element.href}"')
        if element.focused:
            bits.append("focused=true")
        lines.append("  " + " ".join(bits))
    return "\n".join(lines) if lines else "  (no interactive elements found)"


def _viewport_summary(state) -> str:
    if state.page_height <= 0:
        return "scroll position unknown"
    scrolled_pct = min(100, round(100 * state.scroll_y / state.page_height))
    return f"scrolled to y={state.scroll_y} of {state.page_height} ({scrolled_pct}% down the page)"


# Actions after which an unchanged page is expected/normal (waiting for a
# slow load, or scrolling past the end of the page) rather than a sign the
# agent is stuck — safe to auto-retry without spending an LLM call.
_SETTLING_ACTIONS = {"wait", "scroll"}
MAX_AUTO_SKIPS = 3


def _hash_state(state) -> str:
    element_fingerprint = "|".join(f"{e.id}:{e.text}" for e in state.interactive_elements)
    payload = f"{state.url}::{element_fingerprint}::{state.accessibility_tree}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WebAgentReasoner:
    """The Executor in a lightweight Planner/Executor/Verifier split: it
    still runs the step-by-step ReAct loop, but consults a PlannerAgent's
    subtask breakdown as guidance and routes task_complete claims through a
    VerifierAgent before accepting them. Planner/Verifier are auto-built
    from `settings` for real (production) use; they're left off when a test
    supplies its own `router` so existing scripted-router tests aren't
    forced into extra, unscripted LLM calls."""

    def __init__(
        self,
        settings: Settings,
        router: LLMRouter | None = None,
        planner: "PlannerAgent | None" = None,
        verifier: "VerifierAgent | None" = None,
    ):
        self.settings = settings
        self.router = router or build_router(settings)
        self.planner = planner or (PlannerAgent(settings, router=self.router) if router is None else None)
        self.verifier = verifier or (VerifierAgent(settings, router=self.router) if router is None else None)

    def _build_prompt(
        self,
        task: str,
        step_n: int,
        max_steps: int,
        memory: TaskMemory,
        state,
        plan: list[str] | None = None,
    ) -> str:
        plan_text = "\n".join(f"{i}. {step}" for i, step in enumerate(plan or [], start=1))
        return f"""Task: {task}
Step: {step_n} of {max_steps}
Suggested plan (guidance, not a rigid script — deviate if the page calls for it):
{plan_text or '(none)'}

Scratchpad:
{memory.scratchpad or '(empty)'}

Facts learned so far:
{memory.facts_text() or '(none yet)'}

Recent actions:
{memory.recent_history_text() or '(none yet)'}

Recent failures:
{memory.failures_text() or '(none)'}

Current URL: {state.url}
Page title: {state.title}
Viewport: {_viewport_summary(state)}

Interactive elements:
{_elements_as_text(state.interactive_elements)}

Accessibility tree:
{state.accessibility_tree or '(empty)'}

OCR text on screenshot (supplementary; DOM/AX above is more reliable):
{state.ocr_text or '(none)'}

What is your next action?"""

    async def run(
        self,
        task: str,
        session: BrowserSession,
        max_steps: int | None = None,
        tracer=None,
    ) -> AgentRun:
        max_steps = min(
            max_steps or self.settings.max_steps,
            self.settings.max_steps,
            MAX_AGENT_STEPS,
        )
        memory = TaskMemory(goal=task)
        steps: list[Step] = []
        start_time = time.monotonic()
        success = False
        result_text = ""
        previous_state_hash: str | None = None
        previous_action_name: str | None = None
        unchanged_streak = 0
        plan = await self.planner.plan(task) if self.planner else []

        for step_n in range(1, max_steps + 1):
            state = await session.get_page_state()
            state_hash = _hash_state(state)

            if (
                state_hash == previous_state_hash
                and previous_action_name in _SETTLING_ACTIONS
                and unchanged_streak < MAX_AUTO_SKIPS
            ):
                unchanged_streak += 1
                memory.update_scratchpad(
                    f"Page unchanged since last {previous_action_name}; "
                    "auto-waiting instead of re-reasoning."
                )
                action_result = await dispatch_action(session, "wait", {"seconds": 1})
                step = Step(step_n, state.url, "wait", {"seconds": 1}, action_result, "(auto: page unchanged)", 0)
                steps.append(step)
                if tracer:
                    tracer.log_step(
                        step_n,
                        state.url,
                        "wait",
                        {"seconds": 1},
                        action_result,
                        "(auto: page unchanged)",
                        0,
                        elements=[e.id for e in state.interactive_elements],
                    )
                memory.record_action(
                    "wait",
                    "success" if action_result.success else "failed",
                    action_result.new_url,
                    error=None if action_result.success else action_result.error,
                )
                previous_state_hash = state_hash
                previous_action_name = "wait"
                continue

            unchanged_streak = 0
            prompt = self._build_prompt(task, step_n, max_steps, memory, state, plan=plan)
            response = await self._generate_response(prompt, state.screenshot_b64)

            reasoning_text = response.text
            tokens_used = response.tokens_used

            if response.tool_call is None:
                memory.update_scratchpad("No action returned by model; retrying.")
                previous_state_hash = state_hash
                previous_action_name = None
                continue

            action_name = response.tool_call.name
            action_args = response.tool_call.arguments

            if action_name == "task_complete":
                claimed_result = action_args.get("result", "")
                verification = await self._verify_claim(task, claimed_result, state)

                step = Step(
                    step_n,
                    state.url,
                    action_name,
                    action_args,
                    None,
                    reasoning_text,
                    tokens_used,
                )
                steps.append(step)
                if tracer:
                    tracer.log_step(
                        step_n,
                        state.url,
                        action_name,
                        action_args,
                        None,
                        reasoning_text,
                        tokens_used,
                        elements=[e.id for e in state.interactive_elements],
                    )

                if verification is None or verification.verified:
                    result_text = claimed_result
                    success = True
                    break

                memory.update_scratchpad(
                    f"Verifier rejected task_complete claim: {verification.reason}"
                )
                memory.record_action("task_complete", "failed", state.url, error=verification.reason)
                previous_state_hash = state_hash
                previous_action_name = None
                continue

            if action_name == "task_failed":
                result_text = action_args.get("reason", "")
                success = False
                step = Step(
                    step_n,
                    state.url,
                    action_name,
                    action_args,
                    None,
                    reasoning_text,
                    tokens_used,
                )
                steps.append(step)
                if tracer:
                    tracer.log_step(
                        step_n,
                        state.url,
                        action_name,
                        action_args,
                        None,
                        reasoning_text,
                        tokens_used,
                        elements=[e.id for e in state.interactive_elements],
                    )
                break

            validation = validate_action(action_name, action_args, state)
            if validation.ok:
                action_result = await dispatch_action(session, action_name, action_args)
            else:
                memory.update_scratchpad(f"Rejected action: {validation.reason}")
                action_result = ActionResult(
                    success=False,
                    new_url=state.url,
                    error=validation.reason,
                    screenshot_b64=state.screenshot_b64,
                )
            step = Step(
                step_n,
                state.url,
                action_name,
                action_args,
                action_result,
                reasoning_text,
                tokens_used,
            )
            steps.append(step)
            if tracer:
                tracer.log_step(
                    step_n,
                    state.url,
                    action_name,
                    action_args,
                    action_result,
                    reasoning_text,
                    tokens_used,
                    elements=[e.id for e in state.interactive_elements],
                )

            if action_name in _TEXT_RETURNING and action_result.success:
                memory.update_scratchpad(f"{action_name} returned: {action_result.error}")
                memory.record_fact(f"{action_name}_step{step_n}", action_result.error)
            elif not action_result.success:
                memory.update_scratchpad(f"{action_name} failed: {action_result.error}")

            memory.record_action(
                action_name,
                "success" if action_result.success else "failed",
                action_result.new_url,
                error=None if action_result.success else action_result.error,
            )
            if memory.is_looping():
                memory.update_scratchpad(
                    "Detected repeated visits to the same URL; try a different approach."
                )
            if memory.is_repeating_failure(action_name):
                memory.update_scratchpad(
                    f"'{action_name}' has failed repeatedly; try a different action or element."
                )

            previous_state_hash = state_hash
            previous_action_name = action_name

        duration = time.monotonic() - start_time
        final_url = session.page.url if session.page else ""
        return AgentRun(
            task=task,
            success=success,
            result=result_text,
            steps=steps,
            total_actions=len(steps),
            total_tokens=sum(step.tokens_used for step in steps),
            duration_seconds=duration,
            final_url=final_url,
        )

    async def _generate_response(self, prompt: str, screenshot_b64: str) -> LLMResponse:
        return await self.router.plan(SYSTEM_PROMPT, prompt, screenshot_b64, ACTION_SCHEMAS)

    async def _verify_claim(self, task: str, claimed_result: str, state):
        if self.verifier is None:
            return None
        evidence = (
            f"URL: {state.url}\nPage title: {state.title}\n"
            f"Accessibility tree:\n{state.accessibility_tree[:1500] or '(empty)'}"
        )
        return await self.verifier.verify(task, claimed_result, evidence)


if __name__ == "__main__":
    import asyncio

    from agent.llm.base import ToolCall

    class _SmokeRouter:
        async def plan(self, system_prompt, user_prompt, screenshot_b64, tools):
            return LLMResponse(
                text="ready",
                tool_call=ToolCall(name="task_complete", arguments={"result": "smoke ok"}),
                tokens_used=1,
            )

    async def _demo() -> None:
        session = BrowserSession()
        await session.start(headless=True)
        try:
            assert session.page is not None
            await session.page.goto("data:text/html,<title>Smoke</title><h1>Smoke</h1>")
            reasoner = WebAgentReasoner(
                Settings(groq_api_key="fake"),
                router=_SmokeRouter(),
            )
            run = await reasoner.run("Complete the smoke test", session, max_steps=1)
            print(run.success, run.result)
        finally:
            await session.stop()

    asyncio.run(_demo())
