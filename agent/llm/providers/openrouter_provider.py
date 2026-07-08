from __future__ import annotations

import json

import httpx

from agent.llm.base import BaseLLM, LLMProviderError, LLMResponse, ToolCall

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


class OpenRouterLLM(BaseLLM):
    """OpenAI-compatible provider that itself fans out to dozens of models
    on OpenRouter. Counts as one entry in the router's fallback chain."""

    name = "openrouter"

    def __init__(self, api_key: str, model: str, client: httpx.AsyncClient | None = None):
        self.model = model
        self.client = client or httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        )

    async def plan(
        self,
        system_prompt: str,
        user_prompt: str,
        screenshot_b64: str | None,
        tools: list[dict],
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        function_tools = [{"type": "function", "function": schema} for schema in tools]

        try:
            response = await self.client.post(
                _OPENROUTER_URL,
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": function_tools,
                    "tool_choice": "auto",
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMProviderError(str(exc), retryable=True) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(str(exc), retryable=True) from exc

        if response.status_code >= 400:
            retryable = response.status_code in _RETRYABLE_STATUS_CODES
            raise LLMProviderError(
                f"OpenRouter returned {response.status_code}: {response.text[:300]}",
                retryable=retryable,
            )

        payload = response.json()
        message = payload["choices"][0]["message"]
        tool_call = None
        raw_tool_calls = message.get("tool_calls") or []
        if raw_tool_calls:
            call = raw_tool_calls[0]["function"]
            try:
                arguments = json.loads(call.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_call = ToolCall(name=call["name"], arguments=arguments)

        tokens_used = payload.get("usage", {}).get("total_tokens", 0)
        return LLMResponse(
            text=message.get("content") or "", tool_call=tool_call, tokens_used=tokens_used
        )
