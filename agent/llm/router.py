from __future__ import annotations

import asyncio
import logging

from agent.llm.base import BaseLLM, LLMResponse, LLMProviderError

logger = logging.getLogger(__name__)


class AllProvidersFailedError(Exception):
    def __init__(self, errors: list[tuple[str, Exception]]):
        self.errors = errors
        detail = "; ".join(f"{name}: {err}" for name, err in errors)
        super().__init__(f"All LLM providers failed: {detail}")


class LLMRouter:
    """Tries providers in order, retrying each with exponential backoff on
    retryable errors (429/503/timeout) before falling through to the next
    provider. The reasoner talks to this, never to a specific SDK client."""

    def __init__(
        self,
        providers: list[BaseLLM],
        max_retries_per_provider: int = 2,
        base_backoff_seconds: float = 1.0,
    ):
        if not providers:
            raise ValueError("LLMRouter requires at least one provider")
        self.providers = providers
        self.max_retries_per_provider = max_retries_per_provider
        self.base_backoff_seconds = base_backoff_seconds

    async def plan(
        self,
        system_prompt: str,
        user_prompt: str,
        screenshot_b64: str | None,
        tools: list[dict],
    ) -> LLMResponse:
        errors: list[tuple[str, Exception]] = []

        for provider in self.providers:
            try:
                return await self._call_with_retry(
                    provider, system_prompt, user_prompt, screenshot_b64, tools
                )
            except LLMProviderError as exc:
                logger.warning("Provider %s exhausted retries: %s", provider.name, exc)
                errors.append((provider.name, exc))
            except Exception as exc:  # noqa: BLE001 - unexpected provider failure, try next
                logger.warning("Provider %s raised unexpected error: %s", provider.name, exc)
                errors.append((provider.name, exc))

        raise AllProvidersFailedError(errors)

    async def _call_with_retry(
        self,
        provider: BaseLLM,
        system_prompt: str,
        user_prompt: str,
        screenshot_b64: str | None,
        tools: list[dict],
    ) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return await provider.plan(system_prompt, user_prompt, screenshot_b64, tools)
            except LLMProviderError as exc:
                if not exc.retryable or attempt >= self.max_retries_per_provider:
                    raise
                delay = self.base_backoff_seconds * (2**attempt)
                logger.info(
                    "Provider %s failed (attempt %d), retrying in %.1fs: %s",
                    provider.name,
                    attempt + 1,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                attempt += 1
