# agent/config.py
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from agent.llm.providers.gemini_provider import GeminiLLM
from agent.llm.providers.groq_provider import GroqLLM
from agent.llm.providers.ollama_provider import OllamaLLM
from agent.llm.providers.openrouter_provider import OpenRouterLLM
from agent.llm.router import LLMRouter

load_dotenv()

# Ollama last: it's free and always "configured" (no API key needed), but
# weakest at reliable tool-calling, so hosted providers get first shot.
DEFAULT_PROVIDER_ORDER = ("groq", "gemini", "openrouter", "ollama")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    openrouter_api_key: str | None = None
    openrouter_model: str = "openrouter/auto"
    ollama_model: str | None = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
    provider_order: tuple[str, ...] = field(default=DEFAULT_PROVIDER_ORDER)
    max_steps: int = 25
    headless_default: bool = False
    browser_engine: str = "chromium"


def load_settings() -> Settings:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b") or None
    if not any([groq_api_key, gemini_api_key, openrouter_api_key, ollama_model]):
        raise RuntimeError(
            "No LLM provider configured. Set GROQ_API_KEY, GEMINI_API_KEY, "
            "OPENROUTER_API_KEY, or OLLAMA_MODEL in .env"
        )

    order_env = os.environ.get("LLM_PROVIDER_ORDER")
    provider_order = tuple(order_env.split(",")) if order_env else DEFAULT_PROVIDER_ORDER

    return Settings(
        groq_api_key=groq_api_key,
        groq_model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        gemini_api_key=gemini_api_key,
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        openrouter_api_key=openrouter_api_key,
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "openrouter/auto"),
        ollama_model=ollama_model,
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        provider_order=provider_order,
        browser_engine=os.environ.get("BROWSER_ENGINE", "chromium"),
    )


def build_router(settings: Settings) -> LLMRouter:
    """Builds a fallback chain from whichever providers are configured, in
    `settings.provider_order`. Any single provider going down (rate limit,
    outage, no local Ollama) doesn't stop the agent as long as one other
    provider in the chain still works."""
    factories = {
        "groq": lambda: GroqLLM(api_key=settings.groq_api_key, model=settings.groq_model),
        "gemini": lambda: GeminiLLM(api_key=settings.gemini_api_key, model=settings.gemini_model),
        "openrouter": lambda: OpenRouterLLM(
            api_key=settings.openrouter_api_key, model=settings.openrouter_model
        ),
        "ollama": lambda: OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_base_url),
    }
    available = {
        "groq": bool(settings.groq_api_key),
        "gemini": bool(settings.gemini_api_key),
        "openrouter": bool(settings.openrouter_api_key),
        "ollama": bool(settings.ollama_model),
    }

    providers = [factories[name]() for name in settings.provider_order if available.get(name)]
    if not providers:
        raise RuntimeError("No configured LLM provider is available")
    return LLMRouter(providers)


if __name__ == "__main__":
    print(load_settings())
