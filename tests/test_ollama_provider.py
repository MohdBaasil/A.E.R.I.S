import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from ai.ollama_provider import OllamaProvider
from ai.provider import (
    OllamaUnavailableError,
    ProviderModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from config import Config
from core.assistant import Assistant
from core.tool_router import ToolRouter
from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry


def _response(data, status=200):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = data
    if status >= 400:
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=response
        )
    return response


def test_provider_initialization_and_default_settings():
    provider = OllamaProvider()
    assert provider.model == "llama3.2"
    assert provider._base_url == "http://localhost:11434"
    assert provider._timeout == 120
    assert provider._session is not None
    assert provider.requires_startup_health_check is True


def test_chat_success_preserves_roles_and_disables_streaming():
    session = MagicMock()
    session.post.return_value = _response({"message": {"role": "assistant", "content": "Hello there."}})
    provider = OllamaProvider(model="llama3.2", session=session)

    reply = provider.generate([
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "hi"},
    ])

    assert reply == "Hello there."
    args, kwargs = session.post.call_args
    assert args[0] == "http://localhost:11434/api/chat"
    assert kwargs["json"]["model"] == "llama3.2"
    assert kwargs["json"]["messages"] == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "hi"},
    ]
    assert kwargs["json"]["stream"] is False
    assert kwargs["timeout"] == (3, 120)


def test_system_prompt_is_mapped_to_ollama_system_role():
    from core.assistant import SYSTEM_PROMPT_BASE

    provider = OllamaProvider()
    converted = provider._convert_history([
        {"role": "user", "content": SYSTEM_PROMPT_BASE},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "Hi."},
    ])
    assert converted[0] == {"role": "system", "content": SYSTEM_PROMPT_BASE}
    assert converted[1]["role"] == "assistant"
    assert converted[2] == {"role": "user", "content": "Hi."}


def test_system_prompt_requests_native_tools_when_schemas_are_configured():
    from core.assistant import SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_TOOLS

    provider = OllamaProvider()
    provider.configure_tools([{"name": "open_app", "parameters": {"type": "object"}}])
    converted = provider._convert_history([
        {"role": "user", "content": SYSTEM_PROMPT_BASE + SYSTEM_PROMPT_TOOLS},
    ])
    assert "native tools provided" in converted[0]["content"]
    assert "```tool_call" not in converted[0]["content"]


def test_connection_failure_raises_clean_ollama_error():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError("connection refused")
    provider = OllamaProvider(session=session)
    with pytest.raises(OllamaUnavailableError, match="Ollama is unavailable"):
        provider.generate([{"role": "user", "content": "hello"}])


def test_timeout_raises_provider_timeout_error():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ReadTimeout("read timeout")
    provider = OllamaProvider(session=session, timeout_seconds=8)
    with pytest.raises(ProviderTimeoutError):
        provider.generate([{"role": "user", "content": "hello"}])


def test_missing_model_http_404_is_actionable():
    session = MagicMock()
    session.post.return_value = _response({"error": "model not found"}, status=404)
    provider = OllamaProvider(model="llama3.2", session=session)
    with pytest.raises(ProviderModelNotFoundError, match="ollama pull llama3.2"):
        provider.generate([{"role": "user", "content": "hello"}])


def test_malformed_chat_response_raises_provider_response_error():
    session = MagicMock()
    session.post.return_value = _response({"unexpected": True})
    provider = OllamaProvider(session=session)
    with pytest.raises(ProviderResponseError, match="unexpected chat response"):
        provider.generate([{"role": "user", "content": "hello"}])


def test_malformed_json_response_raises_provider_response_error():
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("invalid JSON")
    session.post.return_value = response
    provider = OllamaProvider(session=session)
    with pytest.raises(ProviderResponseError, match="invalid response"):
        provider.generate([{"role": "user", "content": "hello"}])


def test_health_requires_server_and_requested_model():
    session = MagicMock()
    session.get.return_value = _response({"models": [{"name": "llama3.2:latest"}]})
    provider = OllamaProvider(session=session)
    result = provider.health()
    assert result["status"] == "HEALTHY"
    assert result["provider"] == "Ollama"
    session.get.assert_called_once_with(
        "http://localhost:11434/api/tags", timeout=(3, 5)
    )


def test_health_reports_missing_model_without_downloading():
    session = MagicMock()
    session.get.return_value = _response({"models": [{"name": "other:latest"}]})
    provider = OllamaProvider(model="llama3.2", session=session)
    result = provider.health()
    assert result["status"] == "UNAVAILABLE"
    assert "ollama pull llama3.2" in result["reason"]
    session.post.assert_not_called()


def test_health_reports_server_connection_failure_cleanly():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError()
    provider = OllamaProvider(session=session)
    result = provider.health()
    assert result["status"] == "UNAVAILABLE"
    assert "Make sure Ollama is running" in result["reason"]


def test_native_tool_schemas_and_tool_calls_adapt_to_aeris_format():
    session = MagicMock()
    session.post.return_value = _response({
        "message": {
            "role": "assistant",
            "content": "Checking that now.",
            "tool_calls": [{
                "function": {
                    "name": "sum_values",
                    "arguments": {"left": 2, "right": 3},
                }
            }],
        }
    })
    provider = OllamaProvider(session=session)
    schema = {
        "name": "sum_values",
        "description": "Add two values",
        "parameters": {
            "type": "object",
            "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
            "required": ["left", "right"],
        },
        "permission_level": "SAFE",
    }
    provider.configure_tools([schema])

    reply = provider.generate([{"role": "user", "content": "Add two and three"}])

    args, kwargs = session.post.call_args
    assert kwargs["json"]["tools"] == [{
        "type": "function",
        "function": {key: schema[key] for key in ("name", "description", "parameters")},
    }]
    assert reply.startswith("Checking that now.")
    parsed = json.loads(reply.split("```tool_call\n", 1)[1].split("\n```", 1)[0])
    assert parsed == {"tool": "sum_values", "arguments": {"left": 2, "right": 3}}


def test_aeris_tool_transcript_becomes_ollama_tool_exchange():
    provider = OllamaProvider()
    converted = provider._convert_history([
        {"role": "assistant", "content": '```tool_call\n{"tool":"sum_values","arguments":{"left":2,"right":3}}\n```'},
        {"role": "user", "content": "Tool 'sum_values' result:\n```tool_result\n{\"success\": true, \"data\": 5}\n```"},
    ])
    assert converted[0]["role"] == "assistant"
    assert converted[0]["tool_calls"][0]["function"] == {
        "name": "sum_values", "arguments": {"left": 2, "right": 3}
    }
    assert converted[1] == {
        "role": "tool", "name": "sum_values", "content": '{"success": true, "data": 5}'
    }


def test_assistant_tool_flow_executes_native_call_and_returns_summary():
    class SumTool(Tool):
        name = "sum_values"
        description = "Add two integers"
        parameters = {
            "type": "object",
            "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
            "required": ["left", "right"],
        }
        permission_level = PermissionLevel.SAFE

        def execute(self, left=0, right=0, **kwargs):
            return ToolResult.ok(message="Added values", data=left + right)

    session = MagicMock()
    session.post.side_effect = [
        _response({"message": {
            "role": "assistant", "content": "", "tool_calls": [{
                "function": {"name": "sum_values", "arguments": {"left": 2, "right": 3}}
            }]
        }}),
        _response({"message": {"role": "assistant", "content": "The total is five."}}),
    ]
    provider = OllamaProvider(session=session)
    registry = ToolRegistry()
    registry.register(SumTool())
    assistant = Assistant(provider, ToolRouter(registry))

    assert assistant.send("Add two and three") == "The total is five."
    assert session.post.call_count == 2
    second_payload = session.post.call_args_list[1].kwargs["json"]
    assert any(message.get("role") == "tool" for message in second_payload["messages"])


def test_ollama_connection_error_becomes_user_friendly_assistant_message():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError()
    provider = OllamaProvider(session=session)
    assistant = Assistant(provider)
    assert assistant.send("hello") == "Ollama is unavailable. Make sure Ollama is running."


def test_provider_test_reports_ollama_without_api_key(capsys):
    from main import run_provider_test

    config = Config(ai_provider="ollama", ai_api_key="", ai_model="llama3.2")
    provider = MagicMock()
    provider.health.return_value = {
        "status": "HEALTHY", "provider": "Ollama", "model": "llama3.2", "latency": 0.02
    }
    with patch("main.create_provider", return_value=provider):
        result = run_provider_test(config)
    output = capsys.readouterr().out
    assert result == 0
    assert "Provider: Ollama" in output
    assert "Model: llama3.2" in output
    assert "API key: not required" in output
    assert "Status: OK" in output


def test_main_fails_cleanly_when_ollama_is_unavailable(capsys):
    from main import main

    config = Config(ai_provider="ollama", ai_api_key="", ai_model="llama3.2")
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError()
    provider = OllamaProvider(session=session)
    with patch("main.setup_logging"), patch("main.Config.load", return_value=config), \
            patch("main.create_provider", return_value=provider), patch("sys.argv", ["main.py", "--text"]):
        result = main()
    output = capsys.readouterr().out
    assert result == 1
    assert "Ollama is unavailable. Make sure Ollama is running." in output
    assert "Traceback" not in output


def test_ollama_provider_factory_selection_and_gemini_compatibility():
    from ai.provider import GeminiProvider, create_provider

    ollama_config = Config(ai_provider="ollama", ai_api_key="", ai_model="llama3.2")
    assert isinstance(create_provider(ollama_config), OllamaProvider)
    gemini_config = Config(ai_provider="gemini", ai_api_key="fake", ai_model="gemini-3.5-flash-lite")
    from ai.provider import GeminiProvider
    assert isinstance(create_provider(gemini_config), GeminiProvider)


def test_default_keep_alive_is_24h():
    provider = OllamaProvider()
    assert provider.keep_alive == "24h"

    from ai.provider import create_provider
    config = Config(ai_provider="ollama", ai_api_key="", ai_model="llama3.2")
    assert config.ollama_keep_alive == "24h"
    created = create_provider(config)
    assert isinstance(created, OllamaProvider)
    assert created.keep_alive == "24h"


def test_custom_keep_alive_is_respected():
    provider = OllamaProvider(keep_alive="5m")
    assert provider.keep_alive == "5m"

    provider_zero = OllamaProvider(keep_alive="0")
    assert provider_zero.keep_alive == "0"

    from ai.provider import create_provider
    config = Config(ai_provider="ollama", ai_api_key="", ai_model="llama3.2", ollama_keep_alive="1h")
    assert config.ollama_keep_alive == "1h"
    created = create_provider(config)
    assert isinstance(created, OllamaProvider)
    assert created.keep_alive == "1h"


def test_keep_alive_is_included_in_ollama_request_payload():
    session = MagicMock()
    session.post.return_value = _response({"message": {"role": "assistant", "content": "Hi."}})
    provider = OllamaProvider(session=session, keep_alive="12h")

    provider.generate([{"role": "user", "content": "hello"}])
    args, kwargs = session.post.call_args
    assert kwargs["json"]["keep_alive"] == "12h"

    # Also verify streaming payload
    session.post.return_value = MagicMock(
        status_code=200,
        iter_lines=MagicMock(return_value=[json.dumps({"message": {"content": "ok"}}).encode()]),
    )
    list(provider.generate_stream([{"role": "user", "content": "hello"}]))
    stream_args, stream_kwargs = session.post.call_args
    assert stream_kwargs["json"]["keep_alive"] == "12h"


def test_warmup_sends_one_minimal_request_when_enabled():
    session = MagicMock()
    session.post.return_value = _response({"response": ""})
    provider = OllamaProvider(model="qwen2.5:1.5b", session=session, keep_alive="24h")

    success = provider.warmup()
    assert success is True
    assert session.post.call_count == 1
    args, kwargs = session.post.call_args
    assert args[0] == "http://localhost:11434/api/generate"
    assert kwargs["json"] == {"model": "qwen2.5:1.5b", "keep_alive": "24h"}


def test_warmup_failure_does_not_crash_startup():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError("refused")
    provider = OllamaProvider(session=session)
    assert provider.warmup() is False

    session.post.side_effect = None
    session.post.return_value = _response({}, status=500)
    assert provider.warmup() is False

    # Also verify startup in main.py proceeds even if warmup fails
    from main import main
    config = Config(ai_provider="ollama", ai_api_key="", ai_model="llama3.2", ollama_warmup=True)
    provider_mock = MagicMock(spec=OllamaProvider)
    provider_mock.requires_startup_health_check = True
    provider_mock.health.return_value = {"status": "HEALTHY"}
    provider_mock.warmup.side_effect = RuntimeError("Warmup crash")

    with patch("main.setup_logging"), patch("main.Config.load", return_value=config), \
            patch("main.create_provider", return_value=provider_mock), \
            patch("builtins.input", side_effect=KeyboardInterrupt):
        result = main()
    assert result == 0
    provider_mock.warmup.assert_called_once()


def test_warmup_does_not_include_conversation_history_or_tools():
    session = MagicMock()
    session.post.return_value = _response({})
    provider = OllamaProvider(model="llama3.2", session=session, keep_alive="24h")
    provider.configure_tools([{"name": "test_tool", "parameters": {"type": "object"}}])

    provider.warmup()

    assert session.post.call_count == 1
    payload = session.post.call_args.kwargs["json"]
    assert "messages" not in payload
    assert "tools" not in payload
    assert payload == {"model": "llama3.2", "keep_alive": "24h"}
