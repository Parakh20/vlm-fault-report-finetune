from __future__ import annotations

import asyncio
import base64

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from agent.llm.base import BaseLLM, LLMProviderError, LLMResponse, ToolCall

_RETRYABLE_CLIENT_STATUS_CODES = {429}


class GeminiLLM(BaseLLM):
    name = "gemini"

    def __init__(self, api_key: str, model: str, client: genai.Client | None = None):
        self.model = model
        self.client = client or genai.Client(api_key=api_key)

    async def plan(
        self,
        system_prompt: str,
        user_prompt: str,
        screenshot_b64: str | None,
        tools: list[dict],
    ) -> LLMResponse:
        parts = [types.Part(text=user_prompt)]
        if screenshot_b64:
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(screenshot_b64), mime_type="image/png"
                )
            )
        contents = [types.Content(role="user", parts=parts)]

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=tools)],
                    system_instruction=system_prompt,
                ),
            )
        except genai_errors.ServerError as exc:
            raise LLMProviderError(str(exc), retryable=True) from exc
        except genai_errors.ClientError as exc:
            status = getattr(exc, "code", None)
            retryable = status in _RETRYABLE_CLIENT_STATUS_CODES
            raise LLMProviderError(str(exc), retryable=retryable) from exc

        response_parts = response.candidates[0].content.parts
        text = next((p.text for p in response_parts if getattr(p, "text", None)), "")
        function_call = next(
            (p.function_call for p in response_parts if getattr(p, "function_call", None)),
            None,
        )
        tool_call = (
            ToolCall(name=function_call.name, arguments=dict(function_call.args or {}))
            if function_call
            else None
        )
        tokens_used = getattr(response.usage_metadata, "total_token_count", 0)
        return LLMResponse(text=text, tool_call=tool_call, tokens_used=tokens_used)
