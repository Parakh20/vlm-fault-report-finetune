from __future__ import annotations

import json
import re

import httpx

from agent.llm.base import BaseLLM, LLMProviderError, LLMResponse, ToolCall

_DEFAULT_BASE_URL = "http://localhost:11434"
# Small/quantized local models often don't populate the native tool_calls
# field reliably and instead emit the call as a JSON object in the message
# content (e.g. {"name": "click", "arguments": {"element_id": "btn_3"}}).
_INLINE_TOOL_CALL_RE = re.compile(r"\{.*\}", re.DOTALL)


class OllamaLLM(BaseLLM):
    """Local fallback tier: no network dependency, no per-call cost, but
    weaker instruction following than hosted models. Used last in the
    default provider order."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=120.0)

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
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": function_tools,
                    "stream": False,
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMProviderError(str(exc), retryable=True) from exc
        except httpx.RequestError as exc:
            # Ollama not running / unreachable — retrying the same call
            # won't help, move straight to the next provider (if any).
            raise LLMProviderError(str(exc), retryable=False) from exc

        if response.status_code >= 400:
            retryable = response.status_code in (429, 500, 502, 503)
            raise LLMProviderError(
                f"Ollama returned {response.status_code}: {response.text[:300]}",
                retryable=retryable,
            )

        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content") or ""

        tool_call = self._extract_tool_call(message, content)
        tokens_used = payload.get("prompt_eval_count", 0) + payload.get("eval_count", 0)
        return LLMResponse(text=content, tool_call=tool_call, tokens_used=tokens_used)

    def _extract_tool_call(self, message: dict, content: str) -> ToolCall | None:
        raw_tool_calls = message.get("tool_calls") or []
        if raw_tool_calls:
            call = raw_tool_calls[0]["function"]
            arguments = call.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            return ToolCall(name=call["name"], arguments=arguments)

        match = _INLINE_TOOL_CALL_RE.search(content)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if not name or not isinstance(arguments, dict):
            return None
        return ToolCall(name=name, arguments=arguments)
