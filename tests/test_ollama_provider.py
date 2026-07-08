import httpx
import pytest

from agent.llm.base import LLMProviderError
from agent.llm.providers.ollama_provider import OllamaLLM


def _client_with_response(json_body: dict, status_code: int = 200) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=json_body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _client_that_errors() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_plan_parses_native_tool_calls_field():
    # Arrange
    client = _client_with_response(
        {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "click", "arguments": {"element_id": "btn_3"}}}],
            },
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
    )
    provider = OllamaLLM(model="qwen2.5-coder:7b", client=client)

    # Act
    response = await provider.plan("system", "user", None, [])

    # Assert
    assert response.tool_call.name == "click"
    assert response.tool_call.arguments == {"element_id": "btn_3"}
    assert response.tokens_used == 15


@pytest.mark.asyncio
async def test_plan_falls_back_to_parsing_inline_json_in_content():
    # Arrange: model didn't use tool_calls, embedded the call as JSON text instead.
    client = _client_with_response(
        {
            "message": {
                "content": '{\n  "name": "say_hi",\n  "arguments": {\n    "name": "Qwen"\n  }\n}',
            },
            "prompt_eval_count": 100,
            "eval_count": 20,
        }
    )
    provider = OllamaLLM(model="qwen2.5-coder:1.5b", client=client)

    # Act
    response = await provider.plan("system", "user", None, [])

    # Assert
    assert response.tool_call.name == "say_hi"
    assert response.tool_call.arguments == {"name": "Qwen"}


@pytest.mark.asyncio
async def test_plan_returns_none_tool_call_for_plain_text():
    # Arrange
    client = _client_with_response(
        {"message": {"content": "I am thinking about this."}, "prompt_eval_count": 1, "eval_count": 1}
    )
    provider = OllamaLLM(model="qwen2.5-coder:7b", client=client)

    # Act
    response = await provider.plan("system", "user", None, [])

    # Assert
    assert response.tool_call is None
    assert response.text == "I am thinking about this."


@pytest.mark.asyncio
async def test_plan_raises_non_retryable_error_when_ollama_unreachable():
    # Arrange
    provider = OllamaLLM(model="qwen2.5-coder:7b", client=_client_that_errors())

    # Act / Assert
    with pytest.raises(LLMProviderError) as exc_info:
        await provider.plan("system", "user", None, [])
    assert exc_info.value.retryable is False
