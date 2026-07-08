import httpx
import pytest

from agent.llm.base import LLMProviderError
from agent.llm.providers.openrouter_provider import OpenRouterLLM


def _client_with_response(json_body: dict, status_code: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=json_body)

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


@pytest.mark.asyncio
async def test_plan_parses_tool_call_from_openrouter_response():
    # Arrange
    client = _client_with_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "navigate_to", "arguments": '{"url": "https://example.com"}'}}
                        ],
                    }
                }
            ],
            "usage": {"total_tokens": 42},
        }
    )
    provider = OpenRouterLLM(api_key="fake", model="openrouter/auto", client=client)

    # Act
    response = await provider.plan("system", "user", None, [])

    # Assert
    assert response.tool_call.name == "navigate_to"
    assert response.tool_call.arguments == {"url": "https://example.com"}
    assert response.tokens_used == 42


@pytest.mark.asyncio
async def test_plan_returns_text_only_when_no_tool_call():
    # Arrange
    client = _client_with_response(
        {"choices": [{"message": {"content": "just thinking"}}], "usage": {"total_tokens": 5}}
    )
    provider = OpenRouterLLM(api_key="fake", model="openrouter/auto", client=client)

    # Act
    response = await provider.plan("system", "user", None, [])

    # Assert
    assert response.tool_call is None
    assert response.text == "just thinking"


@pytest.mark.asyncio
async def test_plan_raises_retryable_error_on_429():
    # Arrange
    client = _client_with_response({"error": "rate limited"}, status_code=429)
    provider = OpenRouterLLM(api_key="fake", model="openrouter/auto", client=client)

    # Act / Assert
    with pytest.raises(LLMProviderError) as exc_info:
        await provider.plan("system", "user", None, [])
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_plan_raises_non_retryable_error_on_400():
    # Arrange
    client = _client_with_response({"error": "bad request"}, status_code=400)
    provider = OpenRouterLLM(api_key="fake", model="openrouter/auto", client=client)

    # Act / Assert
    with pytest.raises(LLMProviderError) as exc_info:
        await provider.plan("system", "user", None, [])
    assert exc_info.value.retryable is False
