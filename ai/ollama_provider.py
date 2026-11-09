"""Ollama HTTP provider integrated with AERIS's provider and tool contracts."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from typing import Any, Dict, List, Optional

import requests

from ai.provider import (
    AIProvider,
    Message,
    OllamaUnavailableError,
    ProviderError,
    ProviderModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger("aeris.ai.ollama")

_TOOL_CALL_PATTERN = re.compile(r"```tool_call\s*\n(.*?)\n\s*```", re.DOTALL)
_TOOL_RESULT_PATTERN = re.compile(
    r"Tool\s+'([^']+)'\s+result:\s*```tool_result\s*\n(.*?)\n\s*```",
    re.DOTALL,
)


class OllamaProvider(AIProvider):
    """Provider for a locally running Ollama server's ``/api/chat`` endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout_seconds: int = 120,
        session: Optional[requests.Session] = None,
        keep_alive: str = "24h",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = max(1, timeout_seconds)
        self._session = session or requests.Session()
        self._keep_alive = (
            str(keep_alive).strip()
            if (keep_alive is not None and str(keep_alive).strip() != "")
            else "24h"
        )
        self._tools: List[Dict[str, Any]] = []
        self.last_request_seconds = 0.0

    @property
    def keep_alive(self) -> str:
        return self._keep_alive

    @property
    def model(self) -> str:
        return self._model

    @property
    def requires_startup_health_check(self) -> bool:
        return True

    def configure_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Adapt AERIS tool declarations to Ollama's native function tool schema."""
        self._tools = []
        for item in tools:
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            parameters = item.get("parameters")
            if not isinstance(parameters, dict) or not parameters:
                parameters = {"type": "object", "properties": {}}
            self._tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(item.get("description", "")),
                    "parameters": parameters,
                },
            })

    def health(self) -> Dict[str, Any]:
        """Check local server reachability and confirm the selected model is installed."""
        started = time.perf_counter()
        try:
            response = self._session.get(
                f"{self._base_url}/api/tags",
                timeout=(min(3, self._timeout), min(5, self._timeout)),
            )
            latency = round(time.perf_counter() - started, 2)
            if response.status_code != 200:
                return {
                    "status": "UNAVAILABLE",
                    "provider": "Ollama",
                    "model": self._model,
                    "latency": latency,
                    "reason": f"Ollama returned HTTP {response.status_code}.",
                }

            data = response.json()
            models = data.get("models") if isinstance(data, dict) else None
            installed = {
                str(item.get("name") or item.get("model") or "")
                for item in models or []
                if isinstance(item, dict)
            }
            requested = self._model
            requested_tagged = requested if ":" in requested else f"{requested}:latest"
            if requested not in installed and requested_tagged not in installed:
                return {
                    "status": "UNAVAILABLE",
                    "provider": "Ollama",
                    "model": self._model,
                    "latency": latency,
                    "reason": f"Model '{self._model}' is not installed. Run 'ollama pull {self._model}' first.",
                }
            return {
                "status": "HEALTHY",
                "provider": "Ollama",
                "model": self._model,
                "latency": latency,
            }
        except requests.exceptions.Timeout:
            return {
                "status": "UNAVAILABLE",
                "provider": "Ollama",
                "model": self._model,
                "latency": round(time.perf_counter() - started, 2),
                "reason": "Ollama health check timed out.",
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "UNAVAILABLE",
                "provider": "Ollama",
                "model": self._model,
                "latency": round(time.perf_counter() - started, 2),
                "reason": "Ollama is unavailable. Make sure Ollama is running.",
            }
        except (ValueError, requests.exceptions.RequestException):
            return {
                "status": "UNAVAILABLE",
                "provider": "Ollama",
                "model": self._model,
                "latency": round(time.perf_counter() - started, 2),
                "reason": "Ollama returned an invalid health response.",
            }

    def warmup(self) -> bool:
        """Preload the model into memory with the configured keep_alive.

        Returns True if the model was loaded successfully, False otherwise.
        Never raises exceptions and produces no visible output.
        """
        started = time.perf_counter()
        try:
            response = self._session.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "keep_alive": self._keep_alive},
                timeout=(min(3, self._timeout), min(30, self._timeout)),
            )
            elapsed = time.perf_counter() - started
            if response.status_code == 200:
                logger.info(
                    "Ollama model '%s' preloaded successfully (keep_alive: %s, duration: %.2fs).",
                    self._model, self._keep_alive, elapsed,
                )
                return True
            logger.warning(
                "Ollama warmup returned HTTP %s (duration: %.2fs)",
                response.status_code, elapsed,
            )
            return False
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.warning("Ollama warmup failed gracefully: %s (duration: %.2fs)", exc, elapsed)
            return False

    def _build_payload(self, messages: List[Message], stream: bool = False) -> Dict[str, Any]:
        """Build request payload with unified keep_alive and stream settings."""
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_history(messages),
            "stream": stream,
            "keep_alive": self._keep_alive,
        }
        if self._tools:
            payload["tools"] = self._tools
        return payload

    def generate(self, messages: List[Message]) -> str:
        """Generate one response and translate native tool calls to AERIS's router format."""
        payload = self._build_payload(messages, stream=False)

        started = time.perf_counter()
        try:
            response = self._session.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=(min(3, self._timeout), self._timeout),
            )
            if response.status_code == 404:
                raise ProviderModelNotFoundError(
                    f"Ollama model '{self._model}' is not installed. Run 'ollama pull {self._model}' first."
                )
            if response.status_code >= 500:
                raise ProviderUnavailableError(f"Ollama returned HTTP {response.status_code}.")
            response.raise_for_status()
            data = response.json()
        except ProviderError:
            raise
        except requests.exceptions.Timeout as exc:
            raise ProviderTimeoutError("Ollama took too long to respond. Please try again.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnavailableError("Ollama is unavailable. Make sure Ollama is running.") from exc
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            raise ProviderError(f"Ollama returned HTTP {status}.") from exc
        except (requests.exceptions.RequestException, ValueError) as exc:
            raise ProviderResponseError("Ollama returned an invalid response.") from exc
        finally:
            self.last_request_seconds = time.perf_counter() - started

        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise ProviderResponseError("Ollama returned an unexpected chat response.")

        content = message.get("content", "")
        if not isinstance(content, str):
            raise ProviderResponseError("Ollama returned an unexpected chat response.")

        tool_calls = self._normalize_tool_calls(message.get("tool_calls", []))
        if not content.strip() and not tool_calls:
            raise ProviderResponseError("Ollama returned an empty response.")

        if tool_calls:
            blocks = [
                "```tool_call\n" + json.dumps(call, ensure_ascii=False) + "\n```"
                for call in tool_calls
            ]
            return "\n\n".join(part for part in [content.strip(), *blocks] if part)
        return content.strip()

    def generate_stream(
        self,
        messages: List[Message],
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Yield text tokens from Ollama /api/chat stream with tool call translation."""
        payload = self._build_payload(messages, stream=True)
        started = time.perf_counter()
        try:
            response = self._session.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=(min(3, self._timeout), self._timeout),
                stream=True,
            )
            if response.status_code == 404:
                raise ProviderModelNotFoundError(
                    f"Ollama model '{self._model}' is not installed. Run 'ollama pull {self._model}' first."
                )
            if response.status_code >= 500:
                raise ProviderUnavailableError(f"Ollama returned HTTP {response.status_code}.")
            response.raise_for_status()

            buffered_tool_calls: List[Dict[str, Any]] = []
            for line in response.iter_lines():
                if cancel_event and cancel_event.is_set():
                    break
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8") if isinstance(line, bytes) else line)
                except (ValueError, json.JSONDecodeError):
                    continue

                msg = chunk.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if content:
                        yield content

                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        buffered_tool_calls.extend(self._normalize_tool_calls(tool_calls))

            if buffered_tool_calls:
                blocks = [
                    "```tool_call\n" + json.dumps(call, ensure_ascii=False) + "\n```"
                    for call in buffered_tool_calls
                ]
                yield "\n\n" + "\n\n".join(blocks)
        except ProviderError:
            raise
        except requests.exceptions.Timeout as exc:
            raise ProviderTimeoutError("Ollama took too long to respond. Please try again.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnavailableError("Ollama is unavailable. Make sure Ollama is running.") from exc
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            raise ProviderError(f"Ollama returned HTTP {status}.") from exc
        except (requests.exceptions.RequestException, ValueError) as exc:
            raise ProviderResponseError("Ollama returned an invalid response.") from exc
        finally:
            self.last_request_seconds = time.perf_counter() - started

    def _convert_history(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert AERIS history, including its tool transcript, to Ollama chat messages."""
        converted: List[Dict[str, Any]] = []
        for index, item in enumerate(messages):
            role = item.get("role", "user")
            content = str(item.get("content", ""))

            # Assistant bootstrap history stores the system prompt in its first
            # user message. Match the stable assistant identity prefix rather
            # than a stale parenthetical wording so current prompts reach Ollama
            # as a system message too.
            if index == 0 and role == "user" and content.startswith("You are AERIS"):
                system_content = content
                if self._tools:
                    system_content = re.sub(
                        r"- Use this exact format to call a tool:\n```tool_call\n.*?\n```\n",
                        "- Call tools using the native tools provided with this request. Do not write tool calls as text.\n",
                        system_content,
                        count=1,
                        flags=re.DOTALL,
                    )
                converted.append({"role": "system", "content": system_content})
                continue

            if role == "assistant":
                calls: List[Dict[str, Any]] = []
                def remove_call(match: re.Match[str]) -> str:
                    try:
                        parsed = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        return match.group(0)
                    if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
                        args = parsed.get("arguments", {})
                        calls.append({
                            "type": "function",
                            "function": {
                                "name": parsed["tool"],
                                "arguments": args if isinstance(args, dict) else {},
                            },
                        })
                        return ""
                    return match.group(0)

                prose = _TOOL_CALL_PATTERN.sub(remove_call, content).strip()
                assistant_message: Dict[str, Any] = {"role": "assistant", "content": prose}
                if calls:
                    assistant_message["tool_calls"] = calls
                converted.append(assistant_message)
                continue

            if role == "user":
                results: List[Dict[str, str]] = []
                def remove_result(match: re.Match[str]) -> str:
                    results.append({"name": match.group(1), "content": match.group(2).strip()})
                    return ""

                remainder = _TOOL_RESULT_PATTERN.sub(remove_result, content).strip()
                if results:
                    converted.extend({
                        "role": "tool",
                        "name": result["name"],
                        "content": result["content"],
                    } for result in results)
                    if remainder:
                        converted.append({"role": "user", "content": remainder})
                else:
                    converted.append({"role": "user", "content": content})
                continue

            converted.append({"role": role, "content": content})
        return converted

    @staticmethod
    def _normalize_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
        """Normalize Ollama's message.tool_calls payload to AERIS router objects."""
        if not isinstance(raw_calls, list):
            return []
        calls = []
        for item in raw_calls:
            function = item.get("function") if isinstance(item, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            args = function.get("arguments", {})
            if not isinstance(name, str) or not name:
                continue
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({
                "tool": name,
                "arguments": args if isinstance(args, dict) else {},
            })
        return calls
