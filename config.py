"""
Centralized configuration for AERIS.

All environment-dependent values (API keys, model names, provider choice)
are loaded here and nowhere else. Other modules should import from this
file instead of reading os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file, if present.
load_dotenv()


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    ai_provider: str
    ai_api_key: str
    ai_model: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 120
    api_timeout: int = 30
    ai_connect_timeout: int = 5
    ai_read_timeout: int = 12
    ai_retries: int = 1
    ai_retry_backoff: float = 0.5
    ai_circuit_failure_threshold: int = 3
    ai_circuit_cooldown: float = 30.0
    max_tool_rounds: int = 5
    max_history_messages: int = 30
    stt_timeout: int = 4
    stt_phrase_time_limit: int = 12
    stt_pause_threshold: float = 0.55
    stt_non_speaking_duration: float = 0.3
    stt_energy_threshold: int = 300
    stt_dynamic_energy: bool = True
    stt_ambient_duration: float = 0.5
    tts_rate: int = 175
    tts_volume: float = 1.0
    tts_voice: str = ""
    voice_debug: bool = False
    voice_max_history_messages: int = 16
    wake_word_enabled: bool = False
    wake_word: str = "aeris"
    voice_active_window: float = 10.0
    voice_streaming: bool = False
    voice_stream_min_chars: int = 40
    voice_stream_max_chars: int = 180
    voice_stream_flush_timeout: float = 0.8
    mic_device_index: Optional[int] = None
    mic_device_name: str = ""
    mic_auto_select: bool = True
    thinking_threshold: float = 2.0
    log_level: str = "INFO"
    log_max_bytes: int = 2 * 1024 * 1024
    log_backup_count: int = 5
    log_file: str = "aeris.log"
    memory_db_path: str = ""
    memory_max_records: int = 500
    ollama_keep_alive: str = "24h"
    ollama_warmup: bool = True

    @property
    def ai_max_retries(self) -> int:
        """Alias for ai_retries for configuration clarity."""
        return self.ai_retries

    @property
    def voice_streaming_enabled(self) -> bool:
        """Alias for voice_streaming for configuration clarity."""
        return self.voice_streaming

    @staticmethod
    def load() -> "Config":
        provider = os.getenv("AI_PROVIDER", "").strip().lower()
        api_key = os.getenv("AI_API_KEY", "").strip()
        model = os.getenv("AI_MODEL", "").strip()

        if not provider:
            raise ConfigError(
                "AI_PROVIDER is not set. Copy .env.example to .env and fill it in."
            )
        if provider not in {"gemini", "ollama"}:
            raise ConfigError("AI_PROVIDER must be either 'gemini' or 'ollama'.")
        if provider == "gemini" and not api_key:
            raise ConfigError(
                "AI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        if not model:
            raise ConfigError(
                "AI_MODEL is not set. Copy .env.example to .env and fill it in."
            )

        def _get_int(key: str, default: int) -> int:
            val = os.getenv(key, "").strip()
            if not val:
                return default
            try:
                return int(val)
            except ValueError:
                return default

        def _get_optional_int(key: str) -> Optional[int]:
            val = os.getenv(key, "").strip()
            if not val:
                return None
            try:
                return int(val)
            except ValueError:
                return None

        def _get_float(key: str, default: float) -> float:
            val = os.getenv(key, "").strip()
            if not val:
                return default
            try:
                return float(val)
            except ValueError:
                return default

        def _get_bool(key: str, default: bool) -> bool:
            val = os.getenv(key, "").strip().lower()
            if not val:
                return default
            if val in {"1", "true", "yes", "on"}:
                return True
            if val in {"0", "false", "no", "off"}:
                return False
            return default

        api_timeout = _get_int("AI_TIMEOUT", 30)
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
        ollama_timeout = max(1, _get_int("OLLAMA_TIMEOUT", 120))
        if provider == "ollama" and not ollama_base_url.startswith(("http://", "https://")):
            raise ConfigError("OLLAMA_BASE_URL must be an http:// or https:// URL.")
        ai_connect_timeout = max(1, _get_int("AI_CONNECT_TIMEOUT", 5))
        ai_read_timeout = max(1, _get_int("AI_READ_TIMEOUT", 12))
        ai_retries = max(0, _get_int("AI_MAX_RETRIES", _get_int("AI_RETRIES", 1)))
        ai_retry_backoff = max(0.1, _get_float("AI_RETRY_BACKOFF", 0.5))
        ai_circuit_failure_threshold = max(1, _get_int("AI_CIRCUIT_FAILURE_THRESHOLD", 3))
        ai_circuit_cooldown = max(0.1, _get_float("AI_CIRCUIT_COOLDOWN", 30.0))
        max_tool_rounds = _get_int("MAX_TOOL_ROUNDS", 5)
        max_history_messages = _get_int("MAX_HISTORY_MESSAGES", 30)
        stt_timeout = max(1, _get_int("STT_TIMEOUT", 4))
        stt_phrase_time_limit = max(1, _get_int("STT_PHRASE_TIME_LIMIT", 12))
        stt_pause_threshold = max(0.1, _get_float("STT_PAUSE_THRESHOLD", 0.55))
        stt_non_speaking_duration = max(0.1, _get_float("STT_NON_SPEAKING_DURATION", 0.3))
        stt_energy_threshold = max(1, _get_int("STT_ENERGY_THRESHOLD", 300))
        stt_dynamic_energy = _get_bool("STT_DYNAMIC_ENERGY", True)
        stt_ambient_duration = _get_float("STT_AMBIENT_DURATION", 0.5)
        tts_rate = _get_int("TTS_RATE", _get_int("SPEECH_RATE", 175))
        tts_volume = min(1.0, max(0.0, _get_float("TTS_VOLUME", 1.0)))
        tts_voice = os.getenv("TTS_VOICE", "").strip()
        voice_debug = _get_bool("VOICE_DEBUG", False)
        voice_max_history_messages = max(4, _get_int("VOICE_MAX_HISTORY_MESSAGES", 16))
        wake_word_enabled = _get_bool("WAKE_WORD_ENABLED", False)
        wake_word = os.getenv("WAKE_WORD", "aeris").strip() or "aeris"
        voice_active_window = max(1.0, _get_float("VOICE_ACTIVE_WINDOW", 10.0))
        voice_streaming = _get_bool("VOICE_STREAMING_ENABLED", _get_bool("VOICE_STREAMING", False))
        voice_stream_min_chars = max(1, _get_int("VOICE_STREAM_MIN_CHARS", 40))
        voice_stream_max_chars = max(voice_stream_min_chars, _get_int("VOICE_STREAM_MAX_CHARS", 180))
        voice_stream_flush_timeout = max(0.1, _get_float("VOICE_STREAM_FLUSH_TIMEOUT", 0.8))
        mic_device_index = _get_optional_int("MIC_DEVICE_INDEX")
        mic_device_name = os.getenv("MIC_DEVICE_NAME", "").strip()
        mic_auto_select = _get_bool("MIC_AUTO_SELECT", True)
        thinking_threshold = _get_float("THINKING_THRESHOLD", 2.0)
        log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
        log_max_bytes = max(1024, _get_int("LOG_MAX_BYTES", 2 * 1024 * 1024))
        log_backup_count = max(0, _get_int("LOG_BACKUP_COUNT", 5))
        log_file = os.getenv("LOG_FILE", "aeris.log").strip() or "aeris.log"
        memory_db_path = os.getenv("AERIS_MEMORY_DB_PATH", "").strip()
        if not memory_db_path:
            memory_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "aeris_memory.sqlite3")
        memory_max_records = max(1, _get_int("AERIS_MEMORY_MAX_RECORDS", 500))
        ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "24h").strip() or "24h"
        ollama_warmup = _get_bool("OLLAMA_WARMUP", True)

        return Config(
            ai_provider=provider,
            ai_api_key=api_key,
            ai_model=model,
            ollama_base_url=ollama_base_url,
            ollama_timeout=ollama_timeout,
            api_timeout=api_timeout,
            ai_connect_timeout=ai_connect_timeout,
            ai_read_timeout=ai_read_timeout,
            ai_retries=ai_retries,
            ai_retry_backoff=ai_retry_backoff,
            ai_circuit_failure_threshold=ai_circuit_failure_threshold,
            ai_circuit_cooldown=ai_circuit_cooldown,
            max_tool_rounds=max_tool_rounds,
            max_history_messages=max_history_messages,
            stt_timeout=stt_timeout,
            stt_phrase_time_limit=stt_phrase_time_limit,
            stt_pause_threshold=stt_pause_threshold,
            stt_non_speaking_duration=stt_non_speaking_duration,
            stt_energy_threshold=stt_energy_threshold,
            stt_dynamic_energy=stt_dynamic_energy,
            stt_ambient_duration=stt_ambient_duration,
            tts_rate=tts_rate,
            tts_volume=tts_volume,
            tts_voice=tts_voice,
            voice_debug=voice_debug,
            voice_max_history_messages=voice_max_history_messages,
            wake_word_enabled=wake_word_enabled,
            wake_word=wake_word,
            voice_active_window=voice_active_window,
            voice_streaming=voice_streaming,
            voice_stream_min_chars=voice_stream_min_chars,
            voice_stream_max_chars=voice_stream_max_chars,
            voice_stream_flush_timeout=voice_stream_flush_timeout,
            mic_device_index=mic_device_index,
            mic_device_name=mic_device_name,
            mic_auto_select=mic_auto_select,
            thinking_threshold=thinking_threshold,
            log_level=log_level,
            log_max_bytes=log_max_bytes,
            log_backup_count=log_backup_count,
            log_file=log_file,
            memory_db_path=memory_db_path,
            memory_max_records=memory_max_records,
            ollama_keep_alive=ollama_keep_alive,
            ollama_warmup=ollama_warmup,
        )
