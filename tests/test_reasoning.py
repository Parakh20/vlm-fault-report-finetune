import pytest

from agent.browser import BrowserSession
from agent.config import Settings
from agent.llm.base import LLMResponse, ToolCall
from agent.reasoning import WebAgentReasoner
from agent.types import ActionResult, PageState


class _ScriptedRouter:
    """Replays a fixed sequence of LLMResponses, one per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    async def plan(self, system_prompt, user_prompt, screenshot_b64, tools):
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


def _response_for(action_name: str, args: dict, reasoning_text: str = "thinking...") -> LLMResponse:
    return LLMResponse(
        text=reasoning_text,
        tool_call=ToolCall(name=action_name, arguments=args),
        tokens_used=100,
    )


@pytest.mark.asyncio
async def test_run_executes_navigate_then_task_complete(static_server):
    # Arrange
    fake_router = _ScriptedRouter(
        [
            _response_for("navigate_to", {"url": f"{static_server}/sample_page.html"}),
            _response_for("task_complete", {"result": "Found the fixture page."}),
        ]
    )
    settings = Settings(groq_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, router=fake_router)
    session = BrowserSession()
    await session.start(headless=True)

    try:
        # Act
        run = await reasoner.run("Go to the fixture page", session, max_steps=5)

        # Assert
        assert run.success is True
        assert run.result == "Found the fixture page."
        assert run.total_actions == 2
        assert run.final_url.endswith("sample_page.html")
    finally:
        await session.stop()


class _FakeVerifier:
    """Rejects the first N claims, then accepts."""

    def __init__(self, reject_count: int):
        self.reject_count = reject_count
        self.call_count = 0

    async def verify(self, task, claimed_result, final_state_summary):
        from agent.verifier import VerificationResult

        self.call_count += 1
        if self.call_count <= self.reject_count:
            return VerificationResult(verified=False, reason="not enough evidence yet")
        return VerificationResult(verified=True, reason="looks good now")


@pytest.mark.asyncio
async def test_run_retries_after_verifier_rejects_task_complete_claim(static_server):
    # Arrange: model claims done twice before the verifier is satisfied.
    fake_router = _ScriptedRouter(
        [
            _response_for("task_complete", {"result": "premature claim"}),
            _response_for("task_complete", {"result": "final claim"}),
        ]
    )
    settings = Settings(groq_api_key="fake-key")
    reasoner = WebAgentReasoner(
        settings, router=fake_router, verifier=_FakeVerifier(reject_count=1)
    )
    session = BrowserSession()
    await session.start(headless=True)
    await session.page.goto(f"{static_server}/sample_page.html")

    try:
        # Act
        run = await reasoner.run("Do the thing", session, max_steps=5)

        # Assert
        assert run.success is True
        assert run.result == "final claim"
        assert len(run.steps) == 2
    finally:
        await session.stop()


def test_planner_and_verifier_are_auto_wired_for_production_use():
    # Arrange / Act: no explicit router -> real production construction path.
    settings = Settings(groq_api_key="fake-key")
    reasoner = WebAgentReasoner(settings)

    # Assert
    assert reasoner.planner is not None
    assert reasoner.verifier is not None


def test_planner_and_verifier_are_off_by_default_when_router_is_injected():
    # Arrange / Act: explicit router -> test/scripted construction path.
    settings = Settings(groq_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, router=_ScriptedRouter([]))

    # Assert
    assert reasoner.planner is None
    assert reasoner.verifier is None


@pytest.mark.asyncio
async def test_run_rejects_hallucinated_element_id_without_touching_browser(monkeypatch, static_server):
    # Arrange
    fake_router = _ScriptedRouter(
        [
            _response_for("click", {"element_id": "btn_does_not_exist"}),
            _response_for("task_failed", {"reason": "gave up"}),
        ]
    )
    settings = Settings(groq_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, router=fake_router)
    session = BrowserSession()
    await session.start(headless=True)
    await session.page.goto(f"{static_server}/sample_page.html")

    dispatch_calls = []

    async def _spy_dispatch(session, name, args):
        dispatch_calls.append((name, args))
        raise AssertionError("dispatch_action should not be called for an invalid element_id")

    monkeypatch.setattr("agent.reasoning.dispatch_action", _spy_dispatch)

    try:
        # Act
        run = await reasoner.run("Click a button that doesn't exist", session, max_steps=5)

        # Assert
        assert dispatch_calls == []
        assert run.steps[0].action_type == "click"
        assert "btn_does_not_exist" in run.steps[0].action_result.error
        assert run.steps[0].action_result.success is False
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_run_auto_skips_llm_calls_when_page_is_unchanged_after_wait(static_server):
    # Arrange: the model keeps waiting on a page that never changes.
    responses = [_response_for("wait", {"seconds": 0}) for _ in range(10)]
    fake_router = _ScriptedRouter(responses)
    settings = Settings(groq_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, router=fake_router)
    session = BrowserSession()
    await session.start(headless=True)
    await session.page.goto(f"{static_server}/sample_page.html")

    try:
        # Act
        run = await reasoner.run("Wait forever", session, max_steps=8)

        # Assert: 8 steps were taken, but far fewer than 8 real LLM calls,
        # since MAX_AUTO_SKIPS caps consecutive auto-waits before forcing
        # a real reasoning call again.
        assert len(run.steps) == 8
        assert fake_router.call_count < 8
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_run_stops_at_max_steps_if_never_completes(static_server):
    # Arrange
    responses = [_response_for("get_page_text", {}) for _ in range(10)]
    fake_router = _ScriptedRouter(responses)
    settings = Settings(groq_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, router=fake_router)
    session = BrowserSession()
    await session.start(headless=True)
    await session.page.goto(f"{static_server}/sample_page.html")

    try:
        # Act
        run = await reasoner.run("Never-ending task", session, max_steps=3)

        # Assert
        assert run.success is False
        assert len(run.steps) == 3
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_run_records_task_failed(static_server):
    # Arrange
    fake_router = _ScriptedRouter(
        [
            _response_for("task_failed", {"reason": "Login required"}),
        ]
    )
    settings = Settings(groq_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, router=fake_router)
    session = BrowserSession()
    await session.start(headless=True)
    await session.page.goto(f"{static_server}/sample_page.html")

    try:
        # Act
        run = await reasoner.run("Log into a private dashboard", session, max_steps=5)

        # Assert
        assert run.success is False
        assert "Login required" in run.result
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_run_enforces_25_step_hard_cap_even_if_settings_are_higher(monkeypatch):
    # Arrange
    # get_page_text (not a "settling" action) so page-change auto-skip
    # doesn't kick in and this test stays focused on the step-cap behavior.
    responses = [_response_for("get_page_text", {}) for _ in range(30)]
    fake_router = _ScriptedRouter(responses)
    settings = Settings(groq_api_key="fake-key", max_steps=99)
    reasoner = WebAgentReasoner(settings, router=fake_router)

    screenshot_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )

    class _FakePage:
        url = "data:text/html,<title>Fake</title>"

    class _FakeSession:
        page = _FakePage()

        async def get_page_state(self):
            return PageState(
                url=self.page.url,
                title="Fake",
                screenshot_b64=screenshot_b64,
                interactive_elements=[],
                accessibility_tree="",
                scroll_y=0,
                page_height=1,
                dialog_visible=False,
                dialog_text=None,
            )

    async def _fake_dispatch(session, name, args):
        return ActionResult(
            success=True,
            new_url=session.page.url,
            error=None,
            screenshot_b64=screenshot_b64,
        )

    monkeypatch.setattr("agent.reasoning.dispatch_action", _fake_dispatch)

    # Act
    run = await reasoner.run("Never-ending task", _FakeSession(), max_steps=30)

    # Assert
    assert run.success is False
    assert len(run.steps) == 25
    assert fake_router.call_count == 25
