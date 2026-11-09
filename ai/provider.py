"""
AI provider abstraction.

The rest of AERIS talks to `AIProvider.generate(messages)` and
`AIProvider.generate_stream(messages)` only.

Provider Reliability Layer:
--------------------------
1. Structured exception hierarchy:
   - ProviderError (base)
   - ProviderTimeoutError
   - ProviderConnectionError
   - ProviderHTTPError
   - ProviderAuthenticationError (401/403)
   - ProviderRateLimitError (429)
   - ProviderUnavailableError (500/502/503/504 / Circuit open)
   - ProviderResponseError (Malformed or invalid response payload)
2. Fast and bounded (connect_timeout, read_timeout) tuples.
3. Bounded transient retries with exponential backoff.
4. Circuit Breaker protection (CLOSED -> OPEN -> HALF_OPEN).
5. Provider health check diagnostics (`provider.health()`).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, TypedDict

import requests

from ai.circuit_breaker import CircuitBreaker
from config import Config

logger = logging.getLogger("aeris.ai")

_TOOL_CALL_BLOCK = re.compile(r"```tool_call\s*\n(.*?)\n\s*```", re.DOTALL)
_TOOL_RESULT_BLOCK = re.compile(
    r"Tool\s+'([^']+)'\s+result:\s*```tool_result\s*\n(.*?)\n\s*```",
    re.DOTALL,
)


class Message(TypedDict):
    role: str  # "user" or "assistant"
    content: str


# ---------------------------------------------------------------------------
# Structured Exception Hierarchy
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base exception for all AI provider failures."""


class ProviderTimeoutError(ProviderError):
    """Raised when the AI provider request times out."""


class ProviderConnectionError(ProviderError):
    """Raised when connecting to the AI provider fails."""


class OllamaUnavailableError(ProviderConnectionError):
    """Raised when the local Ollama service cannot be reached."""


class ProviderModelNotFoundError(ProviderError):
    """Raised when the selected local model is not installed."""


class ProviderHTTPError(ProviderError):
    """Raised when the AI provider returns an HTTP error status."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProviderAuthenticationError(ProviderHTTPError):
    """Raised on 401 Unauthorized or 403 Forbidden authentication failures."""


class ProviderRateLimitError(ProviderHTTPError):
    """Raised on 429 Too Many Requests / Rate Limit Exceeded."""


class ProviderUnavailableError(ProviderError):
    """Raised when AI service is unavailable (500/502/503/504 or circuit open)."""


class ProviderResponseError(ProviderError):
    """Raised when the AI provider returns a malformed or unexpected response structure."""


# ---------------------------------------------------------------------------
# Base Provider Interface
# ---------------------------------------------------------------------------

class AIProvider(ABC):
    """Common interface every AI backend must implement."""

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """
        Given the full conversation so far (oldest first), return the
        assistant's next reply as plain text.

        Implementations must raise ProviderError subtypes on failure.
        """
        raise NotImplementedError

    def generate_stream(
        self,
        messages: List[Message],
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Yield incremental text tokens when supported by the backend."""
        raise NotImplementedError("This provider does not support streaming.")

    def health(self) -> Dict[str, Any]:
        """Return diagnostic health information about provider connectivity."""
        return {"status": "UNKNOWN", "provider": self.__class__.__name__}

    def configure_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Provide the registered tool schemas to providers that support native tools."""

    def warmup(self) -> bool:
        """Optional warmup/preload for providers that benefit from it."""
        return True

    @property
    def requires_startup_health_check(self) -> bool:
        """Whether main should validate provider health before entering interaction mode."""
        return False

    @property
    def circuit_breaker(self) -> Optional[CircuitBreaker]:
        """Return the attached circuit breaker instance if any."""
        return None


# ---------------------------------------------------------------------------
# Gemini Provider Implementation
# ---------------------------------------------------------------------------

class GeminiProvider(AIProvider):
    """AI provider backed by Google's Gemini API (REST, free-tier friendly)."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    _CONNECT_FRACTION: float = 0.2

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int = 30,
        retries: int = 2,
        session: Optional[requests.Session] = None,
        connect_timeout: Optional[int] = None,
        read_timeout: Optional[int] = None,
        retry_backoff: float = 1.0,
        circuit_failure_threshold: int = 3,
        circuit_cooldown: float = 30.0,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

        _connect = max(1, connect_timeout if connect_timeout is not None else max(5, int(timeout_seconds * self._CONNECT_FRACTION)))
        _read = max(1, read_timeout if read_timeout is not None else timeout_seconds)
        self._timeout_tuple: tuple[int, int] = (_connect, _read)
        self._retries = max(0, retries)
        self._retry_backoff = max(0.1, retry_backoff)
        self._session = session or requests.Session()
        self._tools: List[Dict[str, Any]] = []
        self._cb = circuit_breaker or CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            cooldown_seconds=circuit_cooldown,
        )
        self.last_request_seconds: float = 0.0

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._cb

    def max_wait_seconds(self) -> float:
        """Configured timeout budget for a request that makes no response progress.

        Each attempt can wait up to connect+read. Retries sleep with exponential
        backoff between attempts. Requests' read timeout is an idle-socket timeout,
        so this budget assumes the server stops sending data; it is not a strict
        deadline if a response trickles data continuously.
        """
        attempts = self._retries + 1
        per_attempt = float(self._timeout_tuple[0] + self._timeout_tuple[1])
        backoff_total = sum(self._retry_backoff * (2 ** i) for i in range(max(0, attempts - 1)))
        return per_attempt * attempts + backoff_total

    @property
    def model(self) -> str:
        return self._model

    def configure_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Store registered tools as Gemini function declarations."""
        self._tools = [
            {
                "name": tool["name"],
                "description": str(tool.get("description", "")),
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            }
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str) and tool["name"].strip()
        ]

    def _request_payload(self, contents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a Gemini request, including declarations for registered tools."""
        payload: Dict[str, Any] = {"contents": contents}
        if self._tools:
            payload["tools"] = [{"functionDeclarations": self._tools}]
        return payload

    def _convert_history(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Translate AERIS's fenced tool transcript to Gemini's native content parts."""
        contents: List[Dict[str, Any]] = []
        pending_call_ids: Dict[str, List[str]] = {}

        for message in messages:
            role = "model" if message.get("role") == "assistant" else "user"
            content = str(message.get("content", ""))

            if role == "model":
                parts: List[Dict[str, Any]] = []
                cursor = 0
                for match in _TOOL_CALL_BLOCK.finditer(content):
                    preceding_text = content[cursor:match.start()].strip()
                    if preceding_text:
                        parts.append({"text": preceding_text})
                    cursor = match.end()
                    try:
                        call = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        parts.append({"text": match.group(0)})
                        continue
                    if not isinstance(call, dict) or not isinstance(call.get("tool"), str):
                        parts.append({"text": match.group(0)})
                        continue

                    function_call: Dict[str, Any] = {
                        "name": call["tool"],
                        "args": call.get("arguments", {}) if isinstance(call.get("arguments", {}), dict) else {},
                    }
                    call_id = call.get("id")
                    if isinstance(call_id, str) and call_id:
                        function_call["id"] = call_id
                        pending_call_ids.setdefault(call["tool"], []).append(call_id)
                    parts.append({"functionCall": function_call})

                trailing_text = content[cursor:].strip()
                if trailing_text:
                    parts.append({"text": trailing_text})
                contents.append({"role": role, "parts": parts or [{"text": content}]})
                continue

            results: List[Dict[str, Any]] = []
            for match in _TOOL_RESULT_BLOCK.finditer(content):
                try:
                    result = json.loads(match.group(2))
                except json.JSONDecodeError:
                    result = {"result": match.group(2)}
                function_response: Dict[str, Any] = {
                    "name": match.group(1),
                    "response": result if isinstance(result, dict) else {"result": result},
                }
                call_ids = pending_call_ids.get(match.group(1), [])
                if call_ids:
                    function_response["id"] = call_ids.pop(0)
                results.append({"functionResponse": function_response})

            if results:
                contents.append({"role": role, "parts": results})
            else:
                contents.append({"role": role, "parts": [{"text": content}]})
        return contents

    def _extract_response(self, data: object) -> str:
        """Convert Gemini text and function calls to Assistant's existing response format."""
        text = self._extract_text(data)
        if not isinstance(data, dict):
            return text
        try:
            candidates = data.get("candidates", [])
            parts = candidates[0].get("content", {}).get("parts", [])
            calls = []
            for part in parts:
                call = part.get("functionCall") if isinstance(part, dict) else None
                if isinstance(call, dict) and isinstance(call.get("name"), str):
                    arguments = call.get("args", {})
                    calls.append({
                        "tool": call["name"],
                        "arguments": arguments if isinstance(arguments, dict) else {},
                        **({"id": call["id"]} if isinstance(call.get("id"), str) and call["id"] else {}),
                    })
            if not calls:
                return text
            blocks = [
                "```tool_call\n" + json.dumps(call, ensure_ascii=False) + "\n```"
                for call in calls
            ]
            return (text + "\n\n" if text else "") + "\n\n".join(blocks)
        except (AttributeError, IndexError, TypeError):
            return text

    def health(self) -> Dict[str, Any]:
        """
        Check provider health with a fast test ping.
        Does not expose secrets.
        """
        cb_state = self._cb.state
        if cb_state == CircuitBreaker.STATE_OPEN:
            return {
                "status": "UNAVAILABLE",
                "reason": "Circuit breaker OPEN (temporary outage/cooldown)",
                "provider": "Gemini",
                "model": self._model,
                "latency": 0.0,
                "circuit_state": cb_state,
            }

        url = f"{self.BASE_URL}/{self._model}:generateContent"
        test_payload = {"contents": [{"role": "user", "parts": [{"text": "ping"}]}]}
        test_timeout = (min(self._timeout_tuple[0], 4), min(self._timeout_tuple[1], 6))

        started = time.perf_counter()
        try:
            resp = self._session.post(
                url,
                json=test_payload,
                headers={"x-goog-api-key": self._api_key},
                timeout=test_timeout,
            )
            latency = time.perf_counter() - started
            if resp.status_code == 200:
                self._cb.record_success()
                return {
                    "status": "HEALTHY",
                    "latency": round(latency, 2),
                    "provider": "Gemini",
                    "model": self._model,
                    "circuit_state": self._cb.state,
                }
            elif resp.status_code in (401, 403):
                return {
                    "status": "UNAVAILABLE",
                    "reason": f"Authentication failed ({resp.status_code})",
                    "provider": "Gemini",
                    "model": self._model,
                    "latency": round(latency, 2),
                    "circuit_state": self._cb.state,
                }
            elif resp.status_code == 404:
                return {
                    "status": "UNAVAILABLE",
                    "reason": f"Model '{self._model}' not found (404)",
                    "provider": "Gemini",
                    "model": self._model,
                    "latency": round(latency, 2),
                    "circuit_state": self._cb.state,
                }
            elif resp.status_code == 429:
                return {
                    "status": "DEGRADED",
                    "reason": "Rate limit exceeded (429)",
                    "provider": "Gemini",
                    "model": self._model,
                    "latency": round(latency, 2),
                    "circuit_state": self._cb.state,
                }
            elif resp.status_code in (500, 502, 503, 504):
                return {
                    "status": "UNAVAILABLE",
                    "reason": f"Server unavailable ({resp.status_code})",
                    "provider": "Gemini",
                    "model": self._model,
                    "latency": round(latency, 2),
                    "circuit_state": self._cb.state,
                }
            else:
                return {
                    "status": "UNAVAILABLE",
                    "reason": f"HTTP {resp.status_code}",
                    "provider": "Gemini",
                    "model": self._model,
                    "latency": round(latency, 2),
                    "circuit_state": self._cb.state,
                }
        except requests.exceptions.Timeout:
            latency = time.perf_counter() - started
            return {
                "status": "UNAVAILABLE",
                "reason": "Request timed out",
                "provider": "Gemini",
                "model": self._model,
                "latency": round(latency, 2),
                "circuit_state": self._cb.state,
            }
        except requests.exceptions.ConnectionError:
            latency = time.perf_counter() - started
            return {
                "status": "UNAVAILABLE",
                "reason": "Connection failed",
                "provider": "Gemini",
                "model": self._model,
                "latency": round(latency, 2),
                "circuit_state": self._cb.state,
            }
        except Exception as exc:
            latency = time.perf_counter() - started
            return {
                "status": "UNAVAILABLE",
                "reason": f"Unexpected error: {type(exc).__name__}",
                "provider": "Gemini",
                "model": self._model,
                "latency": round(latency, 2),
                "circuit_state": self._cb.state,
            }

    def generate(self, messages: List[Message]) -> str:
        """Send messages to Gemini generateContent endpoint with retries and circuit breaker."""
        if not self._cb.can_execute():
            logger.warning("Circuit breaker is OPEN — fast-failing request.")
            raise ProviderUnavailableError(
                "The AI service is temporarily unavailable. Please try again in a moment."
            )

        url = f"{self.BASE_URL}/{self._model}:generateContent"
        contents = self._convert_history(messages)

        retries = self._retries + 1
        response = None
        last_exception: Optional[Exception] = None
        request_started = time.perf_counter()

        for attempt in range(1, retries + 1):
            if attempt > 1:
                delay = self._retry_backoff * (2 ** (attempt - 2))
                time.sleep(delay)

            try:
                response = self._session.post(
                    url,
                    json=self._request_payload(contents),
                    headers={"x-goog-api-key": self._api_key},
                    timeout=self._timeout_tuple,
                )
                response.raise_for_status()
                # Success
                break
            except requests.exceptions.HTTPError as exc:
                last_exception = exc
                status_code = getattr(exc.response, "status_code", 0)
                if status_code == 400:
                    self._cb.record_failure()
                    logger.error("Gemini request failed: Bad Request (400)")
                    raise ProviderError("The request was malformed (400 Bad Request).") from exc
                elif status_code in (401, 403):
                    self._cb.record_failure()
                    logger.error("Gemini request failed: Unauthorized/Forbidden (%s)", status_code)
                    raise ProviderAuthenticationError(
                        "Authentication failed. Please check your API key configuration.",
                        status_code=status_code,
                    ) from exc
                elif status_code == 404:
                    self._cb.record_failure()
                    logger.error("Gemini request failed: Not Found (404)")
                    raise ProviderError(f"Model '{self._model}' or endpoint not found (404).") from exc
                elif status_code == 429:
                    logger.warning("Gemini request rate limited (429), attempt %d of %d", attempt, retries)
                    if attempt >= retries:
                        self._cb.record_failure()
                        raise ProviderRateLimitError(
                            "Rate limit exceeded. Please wait before sending more requests.",
                            status_code=429,
                        ) from exc
                elif status_code in (500, 502, 503, 504):
                    logger.warning("Gemini service temporary error (%s), attempt %d of %d", status_code, attempt, retries)
                    if attempt >= retries:
                        self._cb.record_failure()
                        raise ProviderUnavailableError(
                            f"AI service is temporarily unavailable ({status_code}). Please try again later."
                        ) from exc
                else:
                    self._cb.record_failure()
                    logger.error("Gemini HTTP request failed with code %s", status_code)
                    raise ProviderHTTPError(f"Unexpected HTTP error ({status_code}).", status_code=status_code) from exc

            except requests.exceptions.ConnectTimeout:
                last_exception = ProviderTimeoutError("The AI service request timed out.")
                logger.warning("Gemini connect timeout, attempt %d of %d", attempt, retries)
                if attempt >= retries:
                    self._cb.record_failure()
                    raise last_exception

            except requests.exceptions.ReadTimeout:
                last_exception = ProviderTimeoutError("The AI service request timed out.")
                logger.warning("Gemini read timeout, attempt %d of %d", attempt, retries)
                if attempt >= retries:
                    self._cb.record_failure()
                    raise last_exception

            except requests.exceptions.Timeout:
                last_exception = ProviderTimeoutError("The AI service request timed out.")
                logger.warning("Gemini request timeout, attempt %d of %d", attempt, retries)
                if attempt >= retries:
                    self._cb.record_failure()
                    raise last_exception

            except requests.exceptions.ConnectionError:
                last_exception = ProviderConnectionError(
                    "Could not connect to the AI service. Please check your internet connection."
                )
                logger.warning("Gemini connection error, attempt %d of %d", attempt, retries)
                if attempt >= retries:
                    self._cb.record_failure()
                    raise last_exception

            except requests.exceptions.RequestException as exc:
                last_exception = exc
                self._cb.record_failure()
                logger.error("Gemini request failed: %s", type(exc).__name__)
                raise ProviderError("An unexpected network error occurred.") from exc

        self.last_request_seconds = time.perf_counter() - request_started

        if response is None:
            self._cb.record_failure()
            if isinstance(last_exception, ProviderError):
                raise last_exception
            raise ProviderError("Failed to receive response from AI service.")

        try:
            data = response.json()
        except Exception:
            self._cb.record_failure()
            logger.error("Failed to parse Gemini response JSON")
            raise ProviderResponseError("The AI service returned an unparseable response.")

        text = self._extract_response(data).strip()
        if not text:
            # Check for candidate errors or safety blocks
            candidates = data.get("candidates", []) if isinstance(data, dict) else []
            if candidates and isinstance(candidates[0], dict):
                finish_reason = candidates[0].get("finishReason", "")
                if finish_reason:
                    logger.warning("Gemini finished with reason: %s", finish_reason)
            self._cb.record_failure()
            raise ProviderResponseError("The AI service returned an empty or unexpected response.")

        self._cb.record_success()
        return text

    def generate_stream(
        self,
        messages: List[Message],
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Yield text tokens from Gemini SSE stream with circuit breaker and error mapping."""
        if not self._cb.can_execute():
            logger.warning("Circuit breaker is OPEN — fast-failing streaming request.")
            raise ProviderUnavailableError(
                "The AI service is temporarily unavailable. Please try again in a moment."
            )

        url = f"{self.BASE_URL}/{self._model}:streamGenerateContent?alt=sse"
        contents = self._convert_history(messages)
        response = None
        produced_any_text = False

        try:
            response = self._session.post(
                url,
                json=self._request_payload(contents),
                headers={"x-goog-api-key": self._api_key},
                timeout=self._timeout_tuple,
                stream=True,
            )
            status_code = response.status_code
            if status_code == 400:
                self._cb.record_failure()
                logger.error("Gemini streaming request failed: Bad Request (400)")
                raise ProviderError("The request was malformed (400 Bad Request).")
            elif status_code in (401, 403):
                self._cb.record_failure()
                logger.error("Gemini streaming request failed: Unauthorized/Forbidden (%s)", status_code)
                raise ProviderAuthenticationError(
                    "Authentication failed. Please check your API key configuration.",
                    status_code=status_code,
                )
            elif status_code == 404:
                self._cb.record_failure()
                logger.error("Gemini streaming request failed: Not Found (404)")
                raise ProviderError(f"Model '{self._model}' or endpoint not found (404).")
            elif status_code == 429:
                self._cb.record_failure()
                logger.error("Gemini streaming request failed: Rate Limit Exceeded (429)")
                raise ProviderRateLimitError(
                    "Rate limit exceeded. Please wait before sending more requests.",
                    status_code=429,
                )
            elif status_code in (500, 502, 503, 504):
                self._cb.record_failure()
                logger.warning("Gemini streaming service temporary error (%s)", status_code)
                raise ProviderUnavailableError(
                    f"AI service is temporarily unavailable ({status_code}). Please try again later."
                )

            response.raise_for_status()

            for raw_line in response.iter_lines(decode_unicode=True):
                if cancel_event and cancel_event.is_set():
                    break
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("Ignoring malformed Gemini streaming event.")
                    continue
                text = self._extract_response(data)
                if text:
                    produced_any_text = True
                    yield text

            if produced_any_text or (cancel_event and cancel_event.is_set()):
                self._cb.record_success()

        except requests.exceptions.Timeout as exc:
            self._cb.record_failure()
            raise ProviderTimeoutError("The AI service request timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            self._cb.record_failure()
            raise ProviderConnectionError(
                "Could not connect to the AI service. Please check your internet connection."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            self._cb.record_failure()
            status = getattr(exc.response, "status_code", "unknown")
            raise ProviderHTTPError(f"The AI service returned HTTP error {status}.", status_code=getattr(exc.response, "status_code", None)) from exc
        except requests.exceptions.RequestException as exc:
            self._cb.record_failure()
            raise ProviderError("An unexpected network error occurred.") from exc
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

    @staticmethod
    def _extract_text(data: object) -> str:
        """Safely extract text from a complete or partial Gemini response object."""
        if not isinstance(data, dict):
            return ""
        try:
            candidates = data.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                return ""
            content = candidates[0].get("content")
            if not isinstance(content, dict):
                return ""
            parts = content.get("parts")
            if not isinstance(parts, list):
                return ""
            return "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and "text" in part
            )
        except Exception:
            return ""


def create_provider(config: Config) -> AIProvider:
    """Factory: build the configured AI provider from Config."""
    if config.ai_provider == "gemini":
        timeout = getattr(config, "api_timeout", 30)
        retries = getattr(config, "ai_retries", 1)
        retry_backoff = getattr(config, "ai_retry_backoff", 0.5)
        connect_timeout = getattr(config, "ai_connect_timeout", None)
        read_timeout = getattr(config, "ai_read_timeout", None)
        failure_threshold = getattr(config, "ai_circuit_failure_threshold", 3)
        cooldown = getattr(config, "ai_circuit_cooldown", 30.0)

        return GeminiProvider(
            api_key=config.ai_api_key,
            model=config.ai_model,
            timeout_seconds=timeout,
            retries=retries,
            retry_backoff=retry_backoff,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            circuit_failure_threshold=failure_threshold,
            circuit_cooldown=cooldown,
        )

    if config.ai_provider == "ollama":
        from ai.ollama_provider import OllamaProvider

        return OllamaProvider(
            base_url=getattr(config, "ollama_base_url", "http://localhost:11434"),
            model=config.ai_model,
            timeout_seconds=getattr(config, "ollama_timeout", 120),
            keep_alive=getattr(config, "ollama_keep_alive", "24h"),
        )

    raise ProviderError(
        f"Unknown AI_PROVIDER '{config.ai_provider}'. Supported providers: gemini, ollama"
    )
