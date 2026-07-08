import pytest

from agent.llm.base import BaseLLM, LLMProviderError, LLMResponse, ToolCall
from agent.llm.router import AllProvidersFailedError, LLMRouter


class _FakeProvider(BaseLLM):
    def __init__(self, name: str, behaviors: list):
        """`behaviors` is a list of either an LLMResponse to return or an
        exception instance to raise, consumed one per call."""
        self.name = name
        self._behaviors = list(behaviors)
        self.call_count = 0

    async def plan(self, system_prompt, user_prompt, screenshot_b64, tools):
        behavior = self._behaviors[self.call_count]
        self.call_count += 1
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


def _ok_response(name: str = "task_complete") -> LLMResponse:
    return LLMResponse(text="done", tool_call=ToolCall(name=name, arguments={}), tokens_used=5)


async def _no_op_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_plan_returns_first_provider_response_when_it_succeeds():
    # Arrange
    provider = _FakeProvider("groq", [_ok_response()])
    router = LLMRouter([provider])

    # Act
    response = await router.plan("system", "user", None, [])

    # Assert
    assert response.tool_call.name == "task_complete"
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_plan_retries_retryable_error_then_succeeds(monkeypatch):
    # Arrange
    monkeypatch.setattr("agent.llm.router.asyncio.sleep", _no_op_sleep)
    provider = _FakeProvider(
        "groq",
        [LLMProviderError("429 rate limited", retryable=True), _ok_response()],
    )
    router = LLMRouter([provider], max_retries_per_provider=2)

    # Act
    response = await router.plan("system", "user", None, [])

    # Assert
    assert response.tool_call.name == "task_complete"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_plan_falls_back_to_next_provider_after_retries_exhausted(monkeypatch):
    # Arrange
    monkeypatch.setattr("agent.llm.router.asyncio.sleep", _no_op_sleep)
    groq = _FakeProvider(
        "groq",
        [
            LLMProviderError("503", retryable=True),
            LLMProviderError("503", retryable=True),
            LLMProviderError("503", retryable=True),
        ],
    )
    gemini = _FakeProvider("gemini", [_ok_response("navigate_to")])
    router = LLMRouter([groq, gemini], max_retries_per_provider=2)

    # Act
    response = await router.plan("system", "user", None, [])

    # Assert
    assert response.tool_call.name == "navigate_to"
    assert groq.call_count == 3  # 1 initial + 2 retries, all exhausted
    assert gemini.call_count == 1


@pytest.mark.asyncio
async def test_plan_skips_retries_for_non_retryable_error():
    # Arrange
    groq = _FakeProvider("groq", [LLMProviderError("bad request", retryable=False)])
    gemini = _FakeProvider("gemini", [_ok_response()])
    router = LLMRouter([groq, gemini], max_retries_per_provider=5)

    # Act
    response = await router.plan("system", "user", None, [])

    # Assert
    assert groq.call_count == 1
    assert response.tool_call.name == "task_complete"


@pytest.mark.asyncio
async def test_plan_raises_when_all_providers_fail(monkeypatch):
    # Arrange
    monkeypatch.setattr("agent.llm.router.asyncio.sleep", _no_op_sleep)
    groq = _FakeProvider("groq", [LLMProviderError("503", retryable=False)])
    gemini = _FakeProvider("gemini", [LLMProviderError("429", retryable=False)])
    router = LLMRouter([groq, gemini])

    # Act / Assert
    with pytest.raises(AllProvidersFailedError) as exc_info:
        await router.plan("system", "user", None, [])
    assert len(exc_info.value.errors) == 2
