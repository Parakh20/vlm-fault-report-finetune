import pytest

from agent.browser import BrowserSession
from agent.config import Settings
from agent.reasoning import WebAgentReasoner
from agent.types import ActionResult, PageState


class _FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class _FakePart:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeCandidate:
    def __init__(self, parts):
        self.content = _FakeContent(parts)


class _FakeUsage:
    total_token_count = 100


class _FakeResponse:
    def __init__(self, parts):
        self.candidates = [_FakeCandidate(parts)]
        self.usage_metadata = _FakeUsage()


class _ScriptedGeminiClient:
    """Replays a fixed sequence of responses, one per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.models = self
        self.call_count = 0

    def generate_content(self, model, contents, config):
        response = self._responses[self.call_count]
        self.call_count += 1
        return response


def _response_for(action_name: str, args: dict, reasoning_text: str = "thinking..."):
    return _FakeResponse(
        [
            _FakePart(text=reasoning_text),
            _FakePart(function_call=_FakeFunctionCall(action_name, args)),
        ]
    )


@pytest.mark.asyncio
async def test_run_executes_navigate_then_task_complete(static_server):
    # Arrange
    fake_client = _ScriptedGeminiClient(
        [
            _response_for("navigate_to", {"url": f"{static_server}/sample_page.html"}),
            _response_for("task_complete", {"result": "Found the fixture page."}),
        ]
    )
    settings = Settings(gemini_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, client=fake_client)
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


@pytest.mark.asyncio
async def test_run_stops_at_max_steps_if_never_completes(static_server):
    # Arrange
    responses = [_response_for("get_page_text", {}) for _ in range(10)]
    fake_client = _ScriptedGeminiClient(responses)
    settings = Settings(gemini_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, client=fake_client)
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
    fake_client = _ScriptedGeminiClient(
        [
            _response_for("task_failed", {"reason": "Login required"}),
        ]
    )
    settings = Settings(gemini_api_key="fake-key")
    reasoner = WebAgentReasoner(settings, client=fake_client)
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
    responses = [_response_for("wait", {"seconds": 0}) for _ in range(30)]
    fake_client = _ScriptedGeminiClient(responses)
    settings = Settings(gemini_api_key="fake-key", max_steps=99)
    reasoner = WebAgentReasoner(settings, client=fake_client)

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
    assert fake_client.call_count == 25
