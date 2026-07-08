from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_call: ToolCall | None
    tokens_used: int


class LLMProviderError(Exception):
    """Raised when a provider call fails.

    retryable=True means the same provider can be retried with backoff
    (429/503/timeout); retryable=False means the router should move
    straight to the next provider without wasting retries here.
    """

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class BaseLLM(ABC):
    name: str

    @abstractmethod
    async def plan(
        self,
        system_prompt: str,
        user_prompt: str,
        screenshot_b64: str | None,
        tools: list[dict],
    ) -> LLMResponse:
        """Ask the model for the next action. `tools` uses the Gemini-style
        function-declaration shape (name/description/parameters at the top
        level); each provider adapts it to its own wire format."""
