"""Gemini native function calls adapt to AERIS's existing ToolRouter flow."""

import json
from unittest.mock import MagicMock

from ai.provider import GeminiProvider
from core.assistant import Assistant
from core.tool_router import ToolRouter
from tools.base import ToolResult
from tools.registry import ToolRegistry
from tools.system import SystemInfoTool


def _response(payload):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


def test_gemini_function_call_executes_and_returns_native_function_response():
    session = MagicMock()
    session.post.side_effect = [
        _response({
            "candidates": [{"content": {"parts": [{
                "functionCall": {
                    "id": "call-1",
                    "name": "system_info",
                    "args": {},
                }
            }]}}]
        }),
        _response({"candidates": [{"content": {"parts": [{"text": "CPU usage is 23%."}]}}]}),
    ]
    provider = GeminiProvider("test-key", "test-model", retries=0, session=session)
    registry = ToolRegistry()
    system_tool = SystemInfoTool()
    system_tool.execute = lambda **kwargs: ToolResult.ok(
        message="System metrics collected.", data={"cpu": {"usage_percent": 23.0}}
    )
    registry.register(system_tool)
    assistant = Assistant(provider, ToolRouter(registry))

    reply = assistant.send("Please use the registered system_info tool to report the CPU.")

    assert reply == "CPU usage is currently 23.0%."
    assert any("```tool_result" in message["content"] for message in assistant.history)

    first_payload = session.post.call_args_list[0].kwargs["json"]
    declaration = first_payload["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == "system_info"
    assert declaration["parameters"] == SystemInfoTool.parameters

    follow_up_contents = session.post.call_args_list[1].kwargs["json"]["contents"]
    model_call = next(
        part["functionCall"]
        for message in follow_up_contents if message["role"] == "model"
        for part in message["parts"] if "functionCall" in part
    )
    tool_result = next(
        part["functionResponse"]
        for message in follow_up_contents if message["role"] == "user"
        for part in message["parts"] if "functionResponse" in part
    )
    assert model_call == {"name": "system_info", "args": {}, "id": "call-1"}
    assert tool_result["name"] == "system_info"
    assert tool_result["id"] == "call-1"
    assert tool_result["response"]["success"] is True
    assert tool_result["response"]["data"]["cpu"]["usage_percent"] == 23.0


def test_gemini_stream_sends_declarations_and_converts_function_call():
    session = MagicMock()
    payload = {
        "candidates": [{"content": {"parts": [{
            "functionCall": {"name": "system_info", "args": {}}
        }]}}]
    }
    session.post.return_value.status_code = 200
    session.post.return_value.iter_lines.return_value = [
        "data: " + json.dumps(payload),
        "data: [DONE]",
    ]
    provider = GeminiProvider("test-key", "test-model", session=session)
    provider.configure_tools([SystemInfoTool().get_schema()])

    chunks = list(provider.generate_stream([{"role": "user", "content": "CPU usage?"}]))

    sent_payload = session.post.call_args.kwargs["json"]
    assert sent_payload["tools"][0]["functionDeclarations"][0]["name"] == "system_info"
    assert len(chunks) == 1
    assert json.loads(chunks[0].split("\n", 1)[1].rsplit("\n", 1)[0]) == {
        "tool": "system_info", "arguments": {}
    }
