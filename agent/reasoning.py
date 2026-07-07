import base64
import time

from google import genai
from google.genai import types

from agent.actions import ACTION_SCHEMAS, _TEXT_RETURNING, dispatch_action
from agent.browser import BrowserSession
from agent.config import Settings
from agent.memory import TaskMemory
from agent.types import AgentRun, Step

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
        lines.append("  " + " ".join(bits))
    return "\n".join(lines) if lines else "  (no interactive elements found)"


class WebAgentReasoner:
    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.client = client or genai.Client(api_key=settings.gemini_api_key)
        self.tools = [types.Tool(function_declarations=ACTION_SCHEMAS)]

    def _build_prompt(
        self,
        task: str,
        step_n: int,
        max_steps: int,
        memory: TaskMemory,
        state,
    ) -> str:
        return f"""Task: {task}
Step: {step_n} of {max_steps}
Scratchpad:
{memory.scratchpad or '(empty)'}

Recent actions:
{memory.recent_history_text() or '(none yet)'}

Current URL: {state.url}
Page title: {state.title}

Interactive elements:
{_elements_as_text(state.interactive_elements)}

Accessibility tree:
{state.accessibility_tree or '(empty)'}

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
        memory = TaskMemory()
        steps: list[Step] = []
        start_time = time.monotonic()
        success = False
        result_text = ""

        for step_n in range(1, max_steps + 1):
            state = await session.get_page_state()
            prompt = self._build_prompt(task, step_n, max_steps, memory, state)
            response = self._generate_response(prompt, state.screenshot_b64)

            parts = response.candidates[0].content.parts
            reasoning_text = next(
                (part.text for part in parts if getattr(part, "text", None)),
                "",
            )
            function_call = next(
                (
                    part.function_call
                    for part in parts
                    if getattr(part, "function_call", None)
                ),
                None,
            )
            tokens_used = getattr(response.usage_metadata, "total_token_count", 0)

            if function_call is None:
                memory.update_scratchpad("No action returned by model; retrying.")
                continue

            action_name = function_call.name
            action_args = dict(function_call.args or {})

            if action_name == "task_complete":
                result_text = action_args.get("result", "")
                success = True
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
                    )
                break

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
                    )
                break

            action_result = await dispatch_action(session, action_name, action_args)
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
                )

            if action_name in _TEXT_RETURNING and action_result.success:
                memory.update_scratchpad(f"{action_name} returned: {action_result.error}")
            elif not action_result.success:
                memory.update_scratchpad(f"{action_name} failed: {action_result.error}")

            memory.record_action(
                action_name,
                "success" if action_result.success else "failed",
                action_result.new_url,
            )
            if memory.is_looping():
                memory.update_scratchpad(
                    "Detected repeated visits to the same URL; try a different approach."
                )

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

    def _generate_response(self, prompt: str, screenshot_b64: str):
        image_part = types.Part.from_bytes(
            data=base64.b64decode(screenshot_b64),
            mime_type="image/png",
        )
        contents = [
            types.Content(role="user", parts=[types.Part(text=prompt), image_part])
        ]
        return self.client.models.generate_content(
            model=self.settings.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=self.tools,
                system_instruction=SYSTEM_PROMPT,
            ),
        )


if __name__ == "__main__":
    import asyncio

    class _SmokeCall:
        name = "task_complete"
        args = {"result": "smoke ok"}

    class _SmokePart:
        def __init__(self, text=None, function_call=None):
            self.text = text
            self.function_call = function_call

    class _SmokeResponse:
        candidates = [
            type(
                "_Candidate",
                (),
                {
                    "content": type(
                        "_Content",
                        (),
                        {"parts": [_SmokePart("ready", _SmokeCall())]},
                    )()
                },
            )()
        ]
        usage_metadata = type("_Usage", (), {"total_token_count": 1})()

    class _SmokeClient:
        models = None

        def __init__(self):
            self.models = self

        def generate_content(self, model, contents, config):
            return _SmokeResponse()

    async def _demo() -> None:
        session = BrowserSession()
        await session.start(headless=True)
        try:
            assert session.page is not None
            await session.page.goto("data:text/html,<title>Smoke</title><h1>Smoke</h1>")
            reasoner = WebAgentReasoner(
                Settings(gemini_api_key="fake"),
                client=_SmokeClient(),
            )
            run = await reasoner.run("Complete the smoke test", session, max_steps=1)
            print(run.success, run.result)
        finally:
            await session.stop()

    asyncio.run(_demo())
