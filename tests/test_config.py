import pytest

from config import Config, ConfigError


def test_load_succeeds_with_all_vars(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_API_KEY", "fake-key")
    monkeypatch.setenv("AI_MODEL", "gemini-2.0-flash")

    config = Config.load()

    assert config.ai_provider == "gemini"
    assert config.ai_api_key == "fake-key"
    assert config.ai_model == "gemini-2.0-flash"


@pytest.mark.parametrize("missing_var", ["AI_PROVIDER", "AI_API_KEY", "AI_MODEL"])
def test_load_fails_when_var_missing(monkeypatch, missing_var):
    env = {
        "AI_PROVIDER": "gemini",
        "AI_API_KEY": "fake-key",
        "AI_MODEL": "gemini-2.0-flash",
    }
    env[missing_var] = ""
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ConfigError):
        Config.load()


def test_load_with_phase5_defaults(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_API_KEY", "fake-key")
    monkeypatch.setenv("AI_MODEL", "gemini-3.5-flash-lite")

    config = Config.load()
    assert config.api_timeout == 30
    assert config.ai_connect_timeout == 5
    assert config.ai_read_timeout == 12
    assert config.ai_retries == 1
    assert config.ai_retry_backoff == 0.5
    assert config.max_tool_rounds == 5
    assert config.max_history_messages == 30
    assert config.stt_timeout == 4
    assert config.stt_phrase_time_limit == 12
    assert config.log_level == "INFO"
    assert config.memory_db_path.endswith("data\\aeris_memory.sqlite3")
    assert config.memory_max_records == 500


def test_load_custom_memory_database_path(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_API_KEY", "")
    monkeypatch.setenv("AI_MODEL", "qwen2.5:1.5b")
    monkeypatch.setenv("AERIS_MEMORY_DB_PATH", "D:/AERIS-data/memory.sqlite3")
    assert Config.load().memory_db_path == "D:/AERIS-data/memory.sqlite3"


def test_load_custom_memory_retention_limit(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_API_KEY", "")
    monkeypatch.setenv("AI_MODEL", "qwen2.5:1.5b")
    monkeypatch.setenv("AERIS_MEMORY_MAX_RECORDS", "75")
    assert Config.load().memory_max_records == 75


def test_load_invalid_memory_retention_limit_uses_safe_minimum(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_API_KEY", "")
    monkeypatch.setenv("AI_MODEL", "qwen2.5:1.5b")
    monkeypatch.setenv("AERIS_MEMORY_MAX_RECORDS", "-4")
    assert Config.load().memory_max_records == 1


def test_load_with_custom_phase5_vars(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_API_KEY", "fake-key")
    monkeypatch.setenv("AI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("AI_TIMEOUT", "45")
    monkeypatch.setenv("MAX_TOOL_ROUNDS", "8")
    monkeypatch.setenv("MAX_HISTORY_MESSAGES", "50")
    monkeypatch.setenv("STT_TIMEOUT", "7")
    monkeypatch.setenv("STT_PHRASE_TIME_LIMIT", "15")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    config = Config.load()
    assert config.api_timeout == 45
    assert config.max_tool_rounds == 8
    assert config.max_history_messages == 50
    assert config.stt_timeout == 7
    assert config.stt_phrase_time_limit == 15
    assert config.log_level == "DEBUG"


def test_load_ollama_without_gemini_api_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_API_KEY", "")
    monkeypatch.setenv("AI_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "75")

    config = Config.load()

    assert config.ai_provider == "ollama"
    assert config.ai_api_key == ""
    assert config.ai_model == "llama3.2"
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.ollama_timeout == 75


def test_load_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "unknown")
    monkeypatch.setenv("AI_API_KEY", "not-needed")
    monkeypatch.setenv("AI_MODEL", "model")
    with pytest.raises(ConfigError, match="AI_PROVIDER must be either"):
        Config.load()


def test_load_rejects_invalid_ollama_url(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "localhost:11434")
    with pytest.raises(ConfigError, match="OLLAMA_BASE_URL"):
        Config.load()


def test_load_ollama_keep_alive_and_warmup_defaults(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_API_KEY", "")
    monkeypatch.setenv("AI_MODEL", "llama3.2")

    config = Config.load()
    assert config.ollama_keep_alive == "24h"
    assert config.ollama_warmup is True


def test_load_ollama_keep_alive_and_warmup_custom(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("AI_API_KEY", "")
    monkeypatch.setenv("AI_MODEL", "llama3.2")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "30m")
    monkeypatch.setenv("OLLAMA_WARMUP", "false")

    config = Config.load()
    assert config.ollama_keep_alive == "30m"
    assert config.ollama_warmup is False
