from __future__ import annotations

import asyncio
import json

import groq

from agent.llm.base import BaseLLM, LLMProviderError, LLMResponse, ToolCall

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


class GroqLLM(BaseLLM):
    name = "groq"

    def __init__(self, api_key: str, model: str, client: groq.Groq | None = None):
        self.model = model
        self.client = client or groq.Groq(api_key=api_key)

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
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                tools=function_tools,
                tool_choice="auto",
            )
        except groq.BadRequestError as exc:
            return LLMResponse(
                text=f"Error: model returned an unparseable tool call ({exc}).",
                tool_call=None,
                tokens_used=0,
            )
        except groq.APIStatusError as exc:
            retryable = exc.status_code in _RETRYABLE_STATUS_CODES
            raise LLMProviderError(str(exc), retryable=retryable) from exc
        except (groq.APITimeoutError, groq.APIConnectionError) as exc:
            raise LLMProviderError(str(exc), retryable=True) from exc

        message = response.choices[0].message
        tool_call = None
        if message.tool_calls:
            call = message.tool_calls[0].function
            try:
                arguments = json.loads(call.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_call = ToolCall(name=call.name, arguments=arguments)

        tokens_used = getattr(response.usage, "total_tokens", 0)
        return LLMResponse(text=message.content or "", tool_call=tool_call, tokens_used=tokens_used)
