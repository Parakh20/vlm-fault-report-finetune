import pytest
from agent.config import load_settings

def test_load_settings_reads_gemini_api_key_from_env(monkeypatch):
    # Arrange
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-123")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    # Act
    settings = load_settings()

    # Assert
    assert settings.gemini_api_key == "fake-key-123"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.max_steps == 25
    assert settings.headless_default is False


def test_load_settings_raises_when_key_missing(monkeypatch):
    # Arrange
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Act / Assert
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        load_settings()
