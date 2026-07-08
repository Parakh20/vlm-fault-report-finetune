import pytest

from agent.config import Settings, build_router, load_settings
from agent.llm.providers.gemini_provider import GeminiLLM
from agent.llm.providers.groq_provider import GroqLLM
from agent.llm.providers.ollama_provider import OllamaLLM
from agent.llm.providers.openrouter_provider import OpenRouterLLM


def test_load_settings_reads_all_provider_config_from_env(monkeypatch):
    # Arrange
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-openrouter-key")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_ORDER", raising=False)

    # Act
    settings = load_settings()

    # Assert
    assert settings.groq_api_key == "fake-groq-key"
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.gemini_api_key == "fake-gemini-key"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.openrouter_api_key == "fake-openrouter-key"
    assert settings.ollama_model == "qwen2.5-coder:7b"
    assert settings.provider_order == ("groq", "gemini", "openrouter", "ollama")
    assert settings.max_steps == 25
    assert settings.headless_default is False


def test_load_settings_works_with_only_one_provider_key(monkeypatch):
    # Arrange
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "")

    # Act
    settings = load_settings()

    # Assert
    assert settings.groq_api_key == "fake-groq-key"
    assert settings.gemini_api_key is None
    assert settings.ollama_model is None


def test_load_settings_raises_when_nothing_is_configured(monkeypatch):
    # Arrange: no hosted keys, and Ollama explicitly disabled (empty model).
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "")

    # Act / Assert
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        load_settings()


def test_load_settings_defaults_to_ollama_when_no_hosted_key_is_set(monkeypatch):
    # Arrange: this is the out-of-the-box experience with no API keys at all.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    # Act
    settings = load_settings()

    # Assert
    assert settings.ollama_model == "qwen2.5-coder:7b"


def test_load_settings_honors_custom_provider_order(monkeypatch):
    # Arrange
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,groq")

    # Act
    settings = load_settings()

    # Assert
    assert settings.provider_order == ("gemini", "groq")


def test_build_router_only_includes_configured_providers():
    # Arrange
    settings = Settings(
        groq_api_key="fake-groq-key",
        gemini_api_key=None,
        openrouter_api_key=None,
        ollama_model=None,
    )

    # Act
    router = build_router(settings)

    # Assert
    assert len(router.providers) == 1
    assert isinstance(router.providers[0], GroqLLM)


def test_build_router_orders_providers_per_settings():
    # Arrange
    settings = Settings(
        groq_api_key="fake-groq-key",
        gemini_api_key="fake-gemini-key",
        openrouter_api_key=None,
        ollama_model=None,
        provider_order=("gemini", "groq"),
    )

    # Act
    router = build_router(settings)

    # Assert
    assert [type(p) for p in router.providers] == [GeminiLLM, GroqLLM]


def test_build_router_includes_openrouter_and_ollama_when_configured():
    # Arrange
    settings = Settings(
        groq_api_key=None,
        gemini_api_key=None,
        openrouter_api_key="fake-openrouter-key",
        ollama_model="qwen2.5-coder:7b",
        provider_order=("openrouter", "ollama"),
    )

    # Act
    router = build_router(settings)

    # Assert
    assert [type(p) for p in router.providers] == [OpenRouterLLM, OllamaLLM]


def test_build_router_raises_when_nothing_is_configured():
    # Arrange
    settings = Settings(
        groq_api_key=None, gemini_api_key=None, openrouter_api_key=None, ollama_model=None
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="No configured LLM provider"):
        build_router(settings)
