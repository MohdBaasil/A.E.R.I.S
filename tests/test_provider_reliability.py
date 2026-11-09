"""
Tests for AERIS Phase 9 Provider Reliability Layer.

Covers:
  - Structured exception hierarchy
  - Timeout / connection failure with retries
  - HTTP error codes (400, 401, 403, 404, 429, 500-504)
  - Malformed / empty response handling
  - Circuit breaker (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
  - Provider health() diagnostic
  - Assistant.send() safe recovery (no crashes, no secret leakage)
  - Assistant.send_stream() safe recovery
  - Voice loop resilience to all provider error types
  - Text mode resilience to provider errors
  - Secret-safety: API keys must not appear in user-facing error messages
  - Local command interception: provider is never called for voice/exit commands
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, call
from typing import List

import pytest
import requests

from ai.circuit_breaker import CircuitBreaker
from ai.provider import (
    GeminiProvider,
    ProviderError,
    ProviderTimeoutError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderResponseError,
    create_provider,
)
from core.assistant import Assistant, format_provider_error
from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(status_code: int = 200, json_body: dict | None = None, side_effect=None):
    """Return a mock requests.Session with configurable response."""
    session = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    if json_body is not None:
        response.json.return_value = json_body
    if side_effect is not None:
        session.post.side_effect = side_effect
    else:
        session.post.return_value = response
        if status_code != 200:
            http_err = requests.exceptions.HTTPError(response=response)
            response.raise_for_status.side_effect = http_err
        else:
            response.raise_for_status.return_value = None
    return session, response


def _ok_body() -> dict:
    return {"candidates": [{"content": {"parts": [{"text": "hello world"}]}}]}


def _gemini(retries: int = 0, session=None, connect_timeout: int = 2, read_timeout: int = 5) -> GeminiProvider:
    s = session or MagicMock()
    return GeminiProvider(
        api_key="test-key",
        model="test-model",
        timeout_seconds=10,
        retries=retries,
        session=s,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        circuit_failure_threshold=3,
        circuit_cooldown=30.0,  # long cooldown so it stays OPEN during test
    )


# ---------------------------------------------------------------------------
# Structured Exception Hierarchy
# ---------------------------------------------------------------------------

class TestProviderExceptionHierarchy:

    def test_provider_error_is_base(self):
        assert issubclass(ProviderTimeoutError, ProviderError)
        assert issubclass(ProviderConnectionError, ProviderError)
        assert issubclass(ProviderHTTPError, ProviderError)
        assert issubclass(ProviderAuthenticationError, ProviderHTTPError)
        assert issubclass(ProviderRateLimitError, ProviderHTTPError)
        assert issubclass(ProviderUnavailableError, ProviderError)
        assert issubclass(ProviderResponseError, ProviderError)

    def test_http_error_carries_status_code(self):
        err = ProviderHTTPError("test", status_code=404)
        assert err.status_code == 404

    def test_auth_error_carries_status_code(self):
        err = ProviderAuthenticationError("bad key", status_code=401)
        assert err.status_code == 401


# ---------------------------------------------------------------------------
# Timeout Handling
# ---------------------------------------------------------------------------

class TestProviderTimeoutHandling:

    def test_timeout_raises_provider_timeout_error(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("read timed out")
        provider = _gemini(retries=0, session=session)
        with pytest.raises(ProviderTimeoutError):
            provider.generate([{"role": "user", "content": "hello"}])

    def test_timeout_retried_then_raises(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("timeout")
        provider = _gemini(retries=2, session=session)
        with pytest.raises(ProviderTimeoutError):
            provider.generate([{"role": "user", "content": "hello"}])
        assert session.post.call_count == 3  # 1 original + 2 retries

    def test_timeout_trips_circuit_breaker(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("timeout")
        provider = _gemini(retries=0, session=session)
        provider._cb.failure_threshold = 1
        with pytest.raises(ProviderTimeoutError):
            provider.generate([{"role": "user", "content": "hi"}])
        assert provider._cb.state == CircuitBreaker.STATE_OPEN

    def test_explicit_timeout_tuple_passed_to_session(self):
        provider = _gemini(connect_timeout=3, read_timeout=8)
        assert provider._timeout_tuple == (3, 8)

    def test_default_timeout_fractions(self):
        provider = GeminiProvider("k", "m", timeout_seconds=30)
        connect, read = provider._timeout_tuple
        assert connect >= 5
        assert read == 30


# ---------------------------------------------------------------------------
# Connection Error Handling
# ---------------------------------------------------------------------------

class TestProviderConnectionError:

    def test_connection_error_raises_provider_connection_error(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.ConnectionError("refused")
        provider = _gemini(retries=0, session=session)
        with pytest.raises(ProviderConnectionError):
            provider.generate([{"role": "user", "content": "hello"}])

    def test_connection_error_retried(self):
        session = MagicMock()
        # Fail first, then succeed
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = _ok_body()
        session.post.side_effect = [
            requests.exceptions.ConnectionError("refused"),
            response,
        ]
        provider = _gemini(retries=1, session=session)
        result = provider.generate([{"role": "user", "content": "hello"}])
        assert result == "hello world"
        assert session.post.call_count == 2


# ---------------------------------------------------------------------------
# HTTP Error Handling
# ---------------------------------------------------------------------------

class TestProviderHTTPErrors:

    def _make_http_err_session(self, status_code: int):
        session = MagicMock()
        response = MagicMock()
        response.status_code = status_code
        http_err = requests.exceptions.HTTPError(response=response)
        response.raise_for_status.side_effect = http_err
        session.post.return_value = response
        return session

    def test_401_raises_auth_error_no_retry(self):
        session = self._make_http_err_session(401)
        provider = _gemini(retries=2, session=session)
        with pytest.raises(ProviderAuthenticationError):
            provider.generate([{"role": "user", "content": "hi"}])
        assert session.post.call_count == 1  # no retries for auth failures

    def test_403_raises_auth_error_no_retry(self):
        session = self._make_http_err_session(403)
        provider = _gemini(retries=2, session=session)
        with pytest.raises(ProviderAuthenticationError):
            provider.generate([{"role": "user", "content": "hi"}])
        assert session.post.call_count == 1

    def test_404_raises_provider_error_no_retry(self):
        session = self._make_http_err_session(404)
        provider = _gemini(retries=2, session=session)
        with pytest.raises(ProviderError):
            provider.generate([{"role": "user", "content": "hi"}])
        assert session.post.call_count == 1

    def test_400_raises_provider_error_no_retry(self):
        session = self._make_http_err_session(400)
        provider = _gemini(retries=2, session=session)
        with pytest.raises(ProviderError):
            provider.generate([{"role": "user", "content": "hi"}])
        assert session.post.call_count == 1

    def test_429_raises_rate_limit_error_after_retries(self):
        session = self._make_http_err_session(429)
        provider = _gemini(retries=2, session=session)
        with pytest.raises(ProviderRateLimitError):
            provider.generate([{"role": "user", "content": "hi"}])
        assert session.post.call_count == 3

    def test_503_raises_unavailable_after_retries(self):
        session = self._make_http_err_session(503)
        provider = _gemini(retries=2, session=session)
        with pytest.raises(ProviderUnavailableError):
            provider.generate([{"role": "user", "content": "hi"}])
        assert session.post.call_count == 3

    def test_500_retried_then_unavailable(self):
        session = self._make_http_err_session(500)
        provider = _gemini(retries=1, session=session)
        with pytest.raises(ProviderUnavailableError):
            provider.generate([{"role": "user", "content": "hi"}])
        assert session.post.call_count == 2


# ---------------------------------------------------------------------------
# Malformed Response Handling
# ---------------------------------------------------------------------------

class TestProviderResponseValidation:

    def test_empty_candidates_raises_response_error(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {"candidates": []}
        session.post.return_value = response
        provider = _gemini(retries=0, session=session)
        with pytest.raises(ProviderResponseError):
            provider.generate([{"role": "user", "content": "hi"}])

    def test_missing_candidates_raises_response_error(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {}
        session.post.return_value = response
        provider = _gemini(retries=0, session=session)
        with pytest.raises(ProviderResponseError):
            provider.generate([{"role": "user", "content": "hi"}])

    def test_empty_text_raises_response_error(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {"candidates": [{"content": {"parts": [{"text": ""}]}}]}
        session.post.return_value = response
        provider = _gemini(retries=0, session=session)
        with pytest.raises(ProviderResponseError):
            provider.generate([{"role": "user", "content": "hi"}])

    def test_extract_text_handles_missing_text(self):
        result = GeminiProvider._extract_text({"candidates": [{"content": {"parts": []}}]})
        assert result == ""

    def test_extract_text_happy_path(self):
        data = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
        assert GeminiProvider._extract_text(data) == "Hello"

    def test_extract_text_non_dict_returns_empty(self):
        assert GeminiProvider._extract_text(None) == ""
        assert GeminiProvider._extract_text("string") == ""
        assert GeminiProvider._extract_text([]) == ""


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=1.0)
        assert cb.state == CircuitBreaker.STATE_CLOSED

    def test_can_execute_when_closed(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=1.0)
        assert cb.can_execute() is True

    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_CLOSED
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN

    def test_open_circuit_blocks_execution(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN
        assert cb.can_execute() is False

    def test_open_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN
        time.sleep(0.3)
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN
        assert cb.can_execute() is True

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        time.sleep(0.3)
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreaker.STATE_CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        time.sleep(0.3)
        assert cb.state == CircuitBreaker.STATE_HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=1.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitBreaker.STATE_CLOSED

    def test_reset_clears_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure()
        assert cb.state == CircuitBreaker.STATE_OPEN
        cb.reset()
        assert cb.state == CircuitBreaker.STATE_CLOSED
        assert cb.can_execute() is True

    def test_get_status_no_secrets(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)
        status = cb.get_status()
        assert "state" in status
        assert "consecutive_failures" in status
        assert "failure_threshold" in status
        assert "cooldown_seconds" in status
        status_str = str(status)
        assert "key" not in status_str.lower()
        assert "api" not in status_str.lower()

    def test_open_circuit_triggers_fast_fail_in_provider(self):
        """Provider must raise ProviderUnavailableError immediately when circuit is OPEN."""
        session = MagicMock()
        provider = _gemini(retries=0, session=session)
        provider._cb.failure_threshold = 1
        provider._cb.record_failure()
        assert provider._cb.state == CircuitBreaker.STATE_OPEN
        with pytest.raises(ProviderUnavailableError):
            provider.generate([{"role": "user", "content": "hi"}])
        # Session must NOT be called when circuit is open
        session.post.assert_not_called()

    def test_thread_safety(self):
        """CircuitBreaker must be thread-safe under concurrent record_failure calls."""
        cb = CircuitBreaker(failure_threshold=100, cooldown_seconds=60.0)
        errors = []

        def fail_many():
            try:
                for _ in range(20):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fail_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert cb.failure_count == 100


# ---------------------------------------------------------------------------
# Provider Health Check
# ---------------------------------------------------------------------------

class TestProviderHealthCheck:

    def test_health_returns_healthy_on_200(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = _ok_body()
        session.post.return_value = response
        provider = _gemini(session=session)
        result = provider.health()
        assert result["status"] == "HEALTHY"
        assert "latency" in result
        assert result["provider"] == "Gemini"

    def test_health_returns_unavailable_on_503(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 503
        session.post.return_value = response
        provider = _gemini(session=session)
        result = provider.health()
        assert result["status"] == "UNAVAILABLE"
        assert "reason" in result

    def test_health_returns_unavailable_on_timeout(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("timed out")
        provider = _gemini(session=session)
        result = provider.health()
        assert result["status"] == "UNAVAILABLE"
        assert result["reason"] == "Request timed out"

    def test_health_fast_fails_when_circuit_open(self):
        session = MagicMock()
        provider = _gemini(session=session)
        provider._cb.failure_threshold = 1
        provider._cb.record_failure()
        result = provider.health()
        assert result["status"] == "UNAVAILABLE"
        assert "Circuit" in result["reason"]
        session.post.assert_not_called()

    def test_health_never_exposes_api_key(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 503
        session.post.return_value = response
        provider = GeminiProvider(
            api_key="SUPER_SECRET_KEY_12345",
            model="test-model",
            session=session,
        )
        result = provider.health()
        result_str = str(result)
        assert "SUPER_SECRET_KEY_12345" not in result_str

    def test_health_returns_degraded_on_429(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 429
        session.post.return_value = response
        provider = _gemini(session=session)
        result = provider.health()
        assert result["status"] == "DEGRADED"

    def test_health_returns_unavailable_on_auth_failure(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 401
        session.post.return_value = response
        provider = _gemini(session=session)
        result = provider.health()
        assert result["status"] == "UNAVAILABLE"


# ---------------------------------------------------------------------------
# format_provider_error()
# ---------------------------------------------------------------------------

class TestFormatProviderError:

    def test_timeout_message(self):
        msg = format_provider_error(ProviderTimeoutError("timed out"))
        assert "trouble reaching the AI service" in msg
        assert "Please try again." in msg

    def test_unavailable_message(self):
        msg = format_provider_error(ProviderUnavailableError("503"))
        assert "unavailable" in msg.lower() or "temporarily" in msg.lower()

    def test_rate_limit_message(self):
        msg = format_provider_error(ProviderRateLimitError("429", status_code=429))
        assert "rate limit" in msg.lower() or "wait" in msg.lower()

    def test_auth_message(self):
        msg = format_provider_error(ProviderAuthenticationError("401", status_code=401))
        assert "authentication" in msg.lower() or "api key" in msg.lower()

    def test_connection_message(self):
        msg = format_provider_error(ProviderConnectionError("refused"))
        assert "connect" in msg.lower()

    def test_response_message(self):
        msg = format_provider_error(ProviderResponseError("bad shape"))
        assert "unexpected" in msg.lower() or "response" in msg.lower()

    def test_base_provider_error_message(self):
        msg = format_provider_error(ProviderError("generic"))
        assert msg  # Must return a non-empty string

    def test_no_secrets_in_messages(self):
        """Error messages must not contain anything that looks like an API key."""
        provider_err = ProviderError("secret_api_key_12345")
        msg = format_provider_error(provider_err)
        assert "secret_api_key_12345" not in msg


# ---------------------------------------------------------------------------
# Assistant.send() recovery
# ---------------------------------------------------------------------------

class TestAssistantSendRecovery:

    def _make_assistant(self, provider: GeminiProvider) -> Assistant:
        return Assistant(provider=provider, tool_router=None)

    def test_timeout_returns_safe_message_not_exception(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("timeout")
        provider = _gemini(retries=0, session=session)
        assistant = self._make_assistant(provider)
        result = assistant.send("hello")
        assert isinstance(result, str)
        assert len(result) > 0
        # Must NOT raise

    def test_connection_error_returns_safe_message(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.ConnectionError("refused")
        provider = _gemini(retries=0, session=session)
        assistant = self._make_assistant(provider)
        result = assistant.send("hello")
        assert isinstance(result, str)
        assert "connect" in result.lower() or "trouble" in result.lower() or "unavailable" in result.lower()

    def test_auth_error_returns_safe_message(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 401
        http_err = requests.exceptions.HTTPError(response=response)
        response.raise_for_status.side_effect = http_err
        session.post.return_value = response
        provider = _gemini(retries=0, session=session)
        assistant = self._make_assistant(provider)
        result = assistant.send("hello")
        assert "authentication" in result.lower() or "api key" in result.lower()

    def test_rate_limit_returns_safe_message(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 429
        http_err = requests.exceptions.HTTPError(response=response)
        response.raise_for_status.side_effect = http_err
        session.post.return_value = response
        provider = _gemini(retries=0, session=session)
        assistant = self._make_assistant(provider)
        result = assistant.send("hello")
        assert "rate" in result.lower() or "wait" in result.lower() or "limit" in result.lower()

    def test_circuit_open_returns_safe_message(self):
        session = MagicMock()
        provider = _gemini(retries=0, session=session)
        provider._cb.failure_threshold = 1
        provider._cb.record_failure()
        assistant = self._make_assistant(provider)
        result = assistant.send("hello")
        assert isinstance(result, str)
        assert len(result) > 0
        session.post.assert_not_called()

    def test_no_api_key_in_error_message(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("timeout")
        provider = GeminiProvider(
            api_key="SUPER_SECRET_DEADBEEF",
            model="test-model",
            retries=0,
            session=session,
        )
        assistant = self._make_assistant(provider)
        result = assistant.send("hello")
        assert "SUPER_SECRET_DEADBEEF" not in result

    def test_partial_failure_does_not_corrupt_history(self):
        """History must remain valid length after a provider failure."""
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("timeout")
        provider = _gemini(retries=0, session=session)
        assistant = self._make_assistant(provider)
        initial_len = len(assistant.history)
        assistant.send("hello")
        # The failed user message should be rolled back
        assert len(assistant.history) == initial_len

    def test_successful_response_committed_to_history(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = _ok_body()
        session.post.return_value = response
        provider = _gemini(retries=0, session=session)
        assistant = self._make_assistant(provider)
        initial_len = len(assistant.history)
        result = assistant.send("hello")
        assert result == "hello world"
        assert len(assistant.history) == initial_len + 2  # user + assistant messages


# ---------------------------------------------------------------------------
# Voice and text loop recovery (integration-style)
# ---------------------------------------------------------------------------

class TestVoiceLoopProviderRecovery:
    """Provider failures must not crash the voice loop or expose secrets."""

    def _make_assistant(self, reply: str | Exception):
        assistant = MagicMock()
        if isinstance(reply, Exception):
            assistant.send.side_effect = reply
        else:
            assistant.send.return_value = reply
        return assistant

    def _make_stt(self, phrases: list):
        stt = MagicMock()
        stt.device_name = "Test Microphone"
        stt.device_index = 0
        stt.listen.side_effect = phrases
        return stt

    def test_provider_timeout_message_spoken_in_voice_mode(self):
        from main import run_voice_mode
        timeout_msg = "The AI service request timed out. Please try again."
        assistant = self._make_assistant(timeout_msg)
        stt = self._make_stt(["hello aeris", "exit"])
        tts = MagicMock()

        run_voice_mode(assistant, stt, tts)

        spoken = [c.args[0] for c in tts.speak.call_args_list]
        assert any("timed out" in s or "trouble" in s or "unavailable" in s or "timed" in s for s in spoken)

    def test_provider_unavailable_voice_loop_continues(self):
        from main import run_voice_mode
        call_count = [0]

        def fake_send(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return "The AI service is temporarily unavailable. Please try again in a moment."
            return "Hi there!"

        assistant = MagicMock()
        assistant.send.side_effect = fake_send
        stt = self._make_stt(["first query", "second query", "exit"])
        tts = MagicMock()

        run_voice_mode(assistant, stt, tts)

        assert assistant.send.call_count == 2
        stt.close.assert_called_once()

    def test_local_commands_skip_provider_even_when_circuit_open(self):
        """Exit/voice-switch commands must never call the provider."""
        from main import run_voice_mode
        session = MagicMock()
        provider = _gemini(retries=0, session=session)
        provider._cb.failure_threshold = 1
        provider._cb.record_failure()

        assistant = MagicMock()
        stt = self._make_stt(["exit"])
        tts = MagicMock()

        run_voice_mode(assistant, stt, tts)

        assistant.send.assert_not_called()
        session.post.assert_not_called()

    def test_no_api_key_spoken_in_voice_error(self):
        from main import run_voice_mode
        # Assistant wraps provider error into a safe message
        safe_msg = "Could not connect to the AI service. Please check your internet connection."
        assistant = self._make_assistant(safe_msg)
        stt = self._make_stt(["hello", "exit"])
        tts = MagicMock()

        run_voice_mode(assistant, stt, tts)

        spoken = [c.args[0] for c in tts.speak.call_args_list]
        all_spoken = " ".join(spoken)
        assert "key" not in all_spoken.lower() or "api" not in all_spoken.lower()


# ---------------------------------------------------------------------------
# Config: New provider reliability fields
# ---------------------------------------------------------------------------

class TestConfigProviderReliabilityFields:

    def test_default_circuit_breaker_values(self):
        config = Config(
            ai_provider="gemini",
            ai_api_key="k",
            ai_model="m",
        )
        assert config.ai_circuit_failure_threshold == 3
        assert config.ai_circuit_cooldown == 30.0

    def test_default_retry_backoff(self):
        config = Config(
            ai_provider="gemini",
            ai_api_key="k",
            ai_model="m",
        )
        assert config.ai_retry_backoff == 0.5

    def test_load_circuit_breaker_env_vars(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.setenv("AI_API_KEY", "fake-key")
        monkeypatch.setenv("AI_MODEL", "test-model")
        monkeypatch.setenv("AI_CIRCUIT_FAILURE_THRESHOLD", "5")
        monkeypatch.setenv("AI_CIRCUIT_COOLDOWN", "60")
        monkeypatch.setenv("AI_RETRY_BACKOFF", "2.0")
        config = Config.load()
        assert config.ai_circuit_failure_threshold == 5
        assert config.ai_circuit_cooldown == 60.0
        assert config.ai_retry_backoff == 2.0

    def test_load_max_retries_via_ai_max_retries(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.setenv("AI_API_KEY", "fake-key")
        monkeypatch.setenv("AI_MODEL", "test-model")
        monkeypatch.setenv("AI_MAX_RETRIES", "4")
        config = Config.load()
        assert config.ai_retries == 4

    def test_create_provider_passes_circuit_breaker_config(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.setenv("AI_API_KEY", "fake-key")
        monkeypatch.setenv("AI_MODEL", "test-model")
        monkeypatch.setenv("AI_CIRCUIT_FAILURE_THRESHOLD", "7")
        monkeypatch.setenv("AI_CIRCUIT_COOLDOWN", "45")
        config = Config.load()
        provider = create_provider(config)
        assert isinstance(provider, GeminiProvider)
        assert provider._cb.failure_threshold == 7
        assert provider._cb.cooldown_seconds == 45.0


# ---------------------------------------------------------------------------
# Phase 9: retry classification, session reuse, bounded wait, CLI diagnostics
# ---------------------------------------------------------------------------

class TestRetryableVsNonRetryable:

    def test_read_timeout_is_retryable(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.ReadTimeout("read timed out")
        provider = _gemini(retries=1, session=session)
        with pytest.raises(ProviderTimeoutError):
            provider.generate([{"role": "user", "content": "hello"}])
        assert session.post.call_count == 2

    def test_connect_timeout_is_retryable(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.ConnectTimeout("connect timed out")
        provider = _gemini(retries=1, session=session)
        with pytest.raises(ProviderTimeoutError):
            provider.generate([{"role": "user", "content": "hello"}])
        assert session.post.call_count == 2

    def test_502_and_504_are_retryable(self):
        for code in (502, 504):
            session = MagicMock()
            response = MagicMock()
            response.status_code = code
            response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=response)
            session.post.return_value = response
            provider = _gemini(retries=1, session=session)
            with pytest.raises(ProviderUnavailableError):
                provider.generate([{"role": "user", "content": "hi"}])
            assert session.post.call_count == 2

    def test_auth_failure_is_not_retryable(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 403
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=response)
        session.post.return_value = response
        provider = _gemini(retries=5, session=session)
        with pytest.raises(ProviderAuthenticationError):
            provider.generate([{"role": "user", "content": "hi"}])
        assert session.post.call_count == 1


class TestBoundedFailureAndSessionReuse:

    def test_maximum_retry_limit_is_honored(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("timeout")
        provider = _gemini(retries=1, session=session)
        with pytest.raises(ProviderTimeoutError):
            provider.generate([{"role": "user", "content": "hello"}])
        assert session.post.call_count == 2

    def test_zero_retries_makes_a_single_attempt(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("timeout")
        provider = _gemini(retries=0, session=session)
        with pytest.raises(ProviderTimeoutError):
            provider.generate([{"role": "user", "content": "hello"}])
        assert session.post.call_count == 1

    def test_create_provider_uses_interactive_timeout_budget(self):
        config = Config(ai_provider="gemini", ai_api_key="k", ai_model="gemini-3.5-flash-lite")
        provider = create_provider(config)
        assert isinstance(provider, GeminiProvider)
        assert provider._timeout_tuple == (5, 12)
        assert provider._retries == 1
        # Two attempts of connect+read plus one backoff interval.
        assert provider.max_wait_seconds() == pytest.approx(34.5)

    @patch("ai.provider.time.sleep", return_value=None)
    def test_timeout_failure_returns_quickly_when_io_is_mocked(self, _sleep):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.Timeout("timeout")
        provider = _gemini(retries=1, session=session, connect_timeout=5, read_timeout=12)
        started = time.perf_counter()
        with pytest.raises(ProviderTimeoutError):
            provider.generate([{"role": "user", "content": "hello"}])
        elapsed = time.perf_counter() - started
        assert elapsed < 1.0
        assert session.post.call_count == 2

    def test_persistent_session_is_reused_across_requests(self):
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = _ok_body()
        session.post.return_value = response
        provider = _gemini(retries=0, session=session)
        assert provider.generate([{"role": "user", "content": "one"}]) == "hello world"
        assert provider.generate([{"role": "user", "content": "two"}]) == "hello world"
        assert provider._session is session
        assert session.post.call_count == 2
        first_session = session.post.call_args_list[0]
        second_session = session.post.call_args_list[1]
        assert first_session is not None and second_session is not None

    def test_api_key_is_sent_in_header_not_url(self):
        secret = "SUPER_SECRET_KEY_12345"
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = _ok_body()
        session.post.return_value = response
        provider = GeminiProvider(secret, "test-model", retries=0, session=session)

        provider.generate([{"role": "user", "content": "hello"}])

        args, kwargs = session.post.call_args
        assert secret not in args[0]
        assert kwargs["headers"]["x-goog-api-key"] == secret

    def test_health_check_keeps_api_key_out_of_url(self):
        secret = "SUPER_SECRET_KEY_12345"
        session = MagicMock()
        response = MagicMock()
        response.status_code = 200
        session.post.return_value = response
        provider = GeminiProvider(secret, "test-model", session=session)

        assert provider.health()["status"] == "HEALTHY"

        args, kwargs = session.post.call_args
        assert secret not in args[0]
        assert kwargs["headers"]["x-goog-api-key"] == secret


class TestTextAndVoiceRecoveryMessages:

    def test_text_mode_prints_safe_message_and_continues(self, capsys):
        from main import run_text_mode

        assistant = MagicMock()
        assistant.send.side_effect = [
            "I'm having trouble reaching the AI service. Please try again.",
            "Hello again.",
        ]
        with patch("builtins.input", side_effect=["hello", "still there", "exit"]):
            run_text_mode(assistant)
        out = capsys.readouterr().out
        assert "I'm having trouble reaching the AI service. Please try again." in out
        assert "Hello again." in out
        assert assistant.send.call_count == 2

    def test_text_mode_unexpected_exception_does_not_crash(self, capsys):
        from main import run_text_mode

        assistant = MagicMock()
        assistant.send.side_effect = RuntimeError("exploded")
        with patch("builtins.input", side_effect=["hello", "exit"]):
            run_text_mode(assistant)
        out = capsys.readouterr().out
        assert "I'm having trouble reaching the AI service. Please try again." in out
        assert "Traceback" not in out

    def test_voice_mode_timeout_message_then_listening_continues(self, capsys):
        from main import run_voice_mode

        assistant = MagicMock()
        assistant.send.return_value = "I'm having trouble reaching the AI service. Please try again."
        stt = MagicMock()
        stt.device_name = "Test Microphone"
        stt.device_index = 0
        stt.listen.side_effect = ["hello", "exit"]
        tts = MagicMock()
        run_voice_mode(assistant, stt, tts)
        spoken = [c.args[0] for c in tts.speak.call_args_list]
        assert any("trouble reaching the AI service" in s for s in spoken)
        out = capsys.readouterr().out
        assert "Listening..." in out
        assert "Traceback" not in out


class TestProviderDiagnosticCommand:

    def test_provider_test_ok_output(self, capsys):
        from main import run_provider_test

        config = Config(
            ai_provider="gemini",
            ai_api_key="SUPER_SECRET_KEY_SHOULD_NOT_APPEAR",
            ai_model="gemini-3.5-flash-lite",
        )
        provider = MagicMock()
        provider.health.return_value = {
            "status": "HEALTHY",
            "provider": "Gemini",
            "model": "gemini-3.5-flash-lite",
            "latency": 3.45,
        }
        with patch("main.create_provider", return_value=provider):
            code = run_provider_test(config)
        out = capsys.readouterr().out
        assert code == 0
        assert "AERIS Provider Test" in out
        assert "Provider: Gemini" in out
        assert "Model: gemini-3.5-flash-lite" in out
        assert "Status: OK" in out
        assert "Latency: 3.45s" in out
        assert "Retries: up to 1" in out
        assert "SUPER_SECRET_KEY_SHOULD_NOT_APPEAR" not in out

    def test_provider_test_failure_is_clean(self, capsys):
        from main import run_provider_test

        config = Config(
            ai_provider="gemini",
            ai_api_key="SUPER_SECRET_KEY_SHOULD_NOT_APPEAR",
            ai_model="gemini-3.5-flash-lite",
        )
        provider = MagicMock()
        provider.health.return_value = {
            "status": "UNAVAILABLE",
            "reason": "Request timed out",
            "provider": "Gemini",
            "model": "gemini-3.5-flash-lite",
            "latency": 4.0,
        }
        with patch("main.create_provider", return_value=provider):
            code = run_provider_test(config)
        out = capsys.readouterr().out
        assert code == 1
        assert "Status: FAILED" in out
        assert "SUPER_SECRET_KEY_SHOULD_NOT_APPEAR" not in out
        assert "Traceback" not in out

