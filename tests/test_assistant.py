import json
from unittest.mock import MagicMock

from ai.provider import AIProvider, ProviderError
from core.assistant import Assistant
from core.confirmation import AutoDenyHandler
from core.tool_router import RouterResult, ToolCall, ToolRouter
from core.tool_setup import create_tool_registry
from tools.base import ToolResult
from tools.permissions import PermissionLevel


class FakeProvider(AIProvider):
    def __init__(self, reply=None, error=None):
        self._reply = reply
        self._error = error

    def generate(self, messages):
        if self._error:
            raise self._error
        return self._reply


class SequenceProvider(AIProvider):
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def generate(self, messages):
        self.requests.append(messages)
        return self.replies.pop(0)


def test_send_returns_provider_reply_and_updates_history():
    assistant = Assistant(FakeProvider(reply="Hello there."))

    reply = assistant.send("Hi AERIS")

    assert reply == "Hello there."
    assert assistant._history[-2]["content"] == "Hi AERIS"
    assert assistant._history[-1]["content"] == "Hello there."


def test_send_handles_provider_error_without_corrupting_history():
    assistant = Assistant(FakeProvider(error=ProviderError("boom")))
    history_before = list(assistant._history)

    reply = assistant.send("Hi AERIS")

    assert reply == "I'm having trouble reaching the AI service. Please try again."
    assert assistant._history == history_before


def test_reset_clears_conversation_back_to_system_state():
    assistant = Assistant(FakeProvider(reply="ok"))
    assistant.send("first message")

    assistant.reset()

    assert len(assistant._history) == 2


def test_clear_history_alias():
    assistant = Assistant(FakeProvider(reply="ok"))
    assistant.send("first message")
    assistant.send("second message")

    assistant.clear_history()

    assert len(assistant.history) == 2
    assert assistant.history[0]["role"] == "user"
    assert assistant.history[1]["role"] == "assistant"


def test_history_growth_and_multi_turn_context():
    assistant = Assistant(FakeProvider(reply="I remember."))
    assistant.send("My name is Alice.")
    assistant.send("What is my name?")

    hist = assistant.history
    # 2 bootstrap messages + 4 conversation messages = 6
    assert len(hist) == 6
    assert hist[2]["content"] == "My name is Alice."
    assert hist[3]["content"] == "I remember."
    assert hist[4]["content"] == "What is my name?"
    assert hist[5]["content"] == "I remember."


def test_history_limit_truncation():
    # max_history_messages = 6 (2 bootstrap + 4 turn messages)
    assistant = Assistant(FakeProvider(reply="ack"), max_history_messages=6)
    for i in range(10):
        assistant.send(f"Message {i}")

    hist = assistant.history
    assert len(hist) <= 6
    # Bootstrap messages remain intact at the top
    assert hist[0]["role"] == "user"
    assert "AERIS" in hist[0]["content"]
    assert hist[1]["role"] == "assistant"
    # The most recent message is present
    assert hist[-2]["content"] == "Message 9"
    assert hist[-1]["content"] == "ack"


def test_history_pruning_keeps_tool_call_and_result_together():
    from core.tool_router import ToolRouter
    from tools.registry import ToolRegistry
    from tools.base import Tool

    class EchoTool(Tool):
        name = "echo_tool"
        description = "Returns a fixed value."
        parameters = {"type": "object", "properties": {}}
        permission_level = PermissionLevel.SAFE

        def execute(self, **kwargs):
            return ToolResult.ok(message="done")

    provider = SequenceProvider([
        '```tool_call\n{"tool":"echo_tool","arguments":{}}\n```',
        "The action completed.",
    ])
    registry = ToolRegistry()
    registry.register(EchoTool())
    assistant = Assistant(provider, ToolRouter(registry), max_history_messages=4)

    assert assistant.send("Run the tool") == "The action completed."
    dialog = assistant.history[2:]
    call_index = next(i for i, item in enumerate(dialog) if item["role"] == "assistant" and "tool_call" in item["content"])
    result_index = next(i for i, item in enumerate(dialog) if item["role"] == "user" and "tool_result" in item["content"])
    assert result_index == call_index + 1
    assert dialog[-1]["content"] == "The action completed."


def test_multi_tool_mixed_results_stay_associated_and_failure_is_not_success():
    from core.tool_router import ToolRouter
    from tools.registry import ToolRegistry
    from tools.base import Tool

    class GoodTool(Tool):
        name = "good_tool"
        description = "Succeeds."
        parameters = {"type": "object", "properties": {}}
        permission_level = PermissionLevel.SAFE

        def execute(self, **kwargs):
            return ToolResult.ok(message="good")

    class BadTool(Tool):
        name = "bad_tool"
        description = "Fails."
        parameters = {"type": "object", "properties": {}}
        permission_level = PermissionLevel.SAFE

        def execute(self, **kwargs):
            return ToolResult.fail(error="TEST_FAILURE", message="bad")

    provider = SequenceProvider([
        '```tool_call\n{"tool":"good_tool","arguments":{}}\n```\n'
        '```tool_call\n{"tool":"bad_tool","arguments":{}}\n```',
        "Both actions succeeded.",
    ])
    registry = ToolRegistry()
    registry.register(GoodTool())
    registry.register(BadTool())
    assistant = Assistant(provider, ToolRouter(registry))

    reply = assistant.send("Run both")

    assert reply == "The good tool operation completed successfully. The bad tool operation failed, so it was not completed."
    tool_result = next(message["content"] for message in provider.requests[1] if "```tool_result" in message["content"])
    assert "Tool 'good_tool' result" in tool_result
    assert '"success": true' in tool_result
    assert "Tool 'bad_tool' result" in tool_result
    assert '"error": "TEST_FAILURE"' in tool_result


def test_system_info_final_answer_uses_tool_values_not_model_inventions():
    metrics = {
        "cpu": {"available": True, "usage_percent": 5.6, "logical_cores": 20, "physical_cores": 14},
        "memory": {"available": True, "used_gb": 11.2, "total_gb": 15.64, "usage_percent": 71.6},
        "disk": {"available": True, "path": "C:\\", "used_gb": 263.95, "total_gb": 283.46, "usage_percent": 93.1},
        "battery": {"available": True, "percent": 51, "plugged_in": True},
        "os": {"available": True, "system": "Windows", "release": "11", "version": "10.0.26200", "machine": "AMD64"},
    }

    for model_query, expected in (
        ("Please call the system_info tool and report CPU.", "CPU usage is currently 5.6%."),
        ("Please call the system_info tool and report RAM.", "RAM usage is currently 71.6% (11.2 GB used of 15.64 GB)."),
    ):
        registry = create_tool_registry()
        registry.get("system_info").execute = MagicMock(
            return_value=ToolResult.ok(message="live metrics", data=metrics)
        )
        provider = SequenceProvider([
            '```tool_call\n{"tool":"system_info","arguments":{}}\n```',
            "Your CPU is 99% and the battery is fully charged.",
        ])
        assistant = Assistant(provider, ToolRouter(registry))

        # The wording exercises the model-originated tool-call path separately
        # from deterministic pre-routing of direct live metric requests.
        assert assistant.send(model_query) == expected
        assert len(provider.requests) == 2

    registry = create_tool_registry()
    registry.get("system_info").execute = MagicMock(
        return_value=ToolResult.ok(message="live metrics", data=metrics)
    )
    provider = SequenceProvider([
        '```tool_call\n{"tool":"system_info","arguments":{}}\n```',
        "The battery is fully charged.",
    ])
    assistant = Assistant(provider, ToolRouter(registry))
    answer = assistant.send("What is my system information?")
    assert "CPU usage is currently 5.6%." in answer
    assert "RAM usage is currently 71.6%" in answer
    assert "Battery: 51%, plugged in." in answer
    assert "battery is fully charged" not in answer.lower()


def test_streamed_system_info_answer_uses_tool_values_not_model_inventions():
    class StreamingProvider(SequenceProvider):
        def generate_stream(self, messages, cancel_event=None):
            reply = self.replies.pop(0)
            midpoint = max(1, len(reply) // 2)
            yield reply[:midpoint]
            yield reply[midpoint:]

    registry = create_tool_registry()
    registry.get("system_info").execute = MagicMock(return_value=ToolResult.ok(
        message="live metrics",
        data={"cpu": {"available": True, "usage_percent": 23.4}},
    ))
    provider = StreamingProvider([
        '```tool_call\n{"tool":"system_info","arguments":{}}\n```',
        "CPU usage is 99% and RAM usage is 100%.",
    ])
    assistant = Assistant(provider, ToolRouter(registry))

    chunks = list(assistant.send_stream("Please call the system_info tool and report CPU."))

    assert "".join(chunks) == "CPU usage is currently 23.4%."
    assert registry.get("system_info").execute.call_count == 1
    assert any("```tool_result" in item["content"] for item in assistant.history)


def test_max_tool_rounds_exhaustion_returns_safe_explanation():
    # Provider always responds with a tool_call
    mock_provider = FakeProvider(
        reply='```tool_call\n{"tool": "dummy", "arguments": {}}\n```'
    )
    from core.tool_router import ToolRouter
    from tools.registry import ToolRegistry
    from core.confirmation import AutoApproveHandler

    router = ToolRouter(ToolRegistry(), confirmation_handler=AutoApproveHandler())
    assistant = Assistant(mock_provider, tool_router=router, max_tool_rounds=3)

    reply = assistant.send("Start endless loop")

    assert "requested tool is unavailable" in reply
    assert len(assistant.history) > 2


def _denied_delete_assistant(tmp_path):
    target = tmp_path / "disposable-test.txt"
    target.write_text("keep this file", encoding="utf-8")
    tool_call = (
        "```tool_call\n"
        + json.dumps({"tool": "delete_file", "arguments": {"path": str(target)}})
        + "\n```"
    )
    provider = SequenceProvider([tool_call, "The file has been deleted."])
    registry = create_tool_registry()
    confirmation = MagicMock()
    confirmation.request_confirmation.return_value = False
    assistant = Assistant(provider, ToolRouter(registry, confirmation_handler=confirmation))
    return target, provider, confirmation, assistant


def test_denied_delete_cannot_be_reported_as_success_and_result_is_sent_to_model(tmp_path):
    target, provider, confirmation, assistant = _denied_delete_assistant(tmp_path)

    reply = assistant.send("Delete disposable-test.txt")

    assert reply == "The deletion was cancelled because you declined permission. The file was not deleted."
    assert target.exists()
    confirmation.request_confirmation.assert_called_once()
    assert confirmation.request_confirmation.call_args.kwargs["permission_level"] == PermissionLevel.DANGEROUS.value
    assert len(provider.requests) == 2
    follow_up = provider.requests[1]
    tool_result = next(message["content"] for message in follow_up if "```tool_result" in message["content"])
    assert '"success": false' in tool_result
    assert '"error": "PERMISSION_DENIED"' in tool_result
    assert "not deleted" in reply.lower()


def test_streamed_denied_delete_cannot_leak_model_success_claim(tmp_path):
    target, provider, confirmation, assistant = _denied_delete_assistant(tmp_path)

    chunks = list(assistant.send_stream("Delete disposable-test.txt"))

    assert chunks == ["The deletion was cancelled because you declined permission. The file was not deleted."]
    assert target.exists()
    confirmation.request_confirmation.assert_called_once()
    follow_up = provider.requests[1]
    assert any('"error": "PERMISSION_DENIED"' in message["content"] for message in follow_up)


def test_tool_failure_outcomes_are_distinguished_deterministically():
    cases = (
        ("delete_file", "PERMISSION_DENIED", "The deletion was cancelled because you declined permission. The file was not deleted."),
        ("move_file", "CONFIRMATION_UNAVAILABLE", "The required confirmation was unavailable, so the action was not performed."),
        ("missing_tool", "UNKNOWN_TOOL", "The requested tool is unavailable, so the action was not performed."),
        ("rename_file", "RENAME_FAILED", "The rename file operation failed, so it was not completed."),
    )
    from core.assistant import _format_tool_outcomes

    for tool_name, error, expected in cases:
        result = RouterResult(
            is_tool_call=True,
            tool_calls=[ToolCall(tool=tool_name)],
            tool_results=[ToolResult.fail(error=error, message="not successful")],
        )
        assert _format_tool_outcomes(result) == expected

    success = RouterResult(
        is_tool_call=True,
        tool_calls=[ToolCall(tool="system_info")],
        tool_results=[ToolResult.ok("metrics retrieved")],
    )
    assert _format_tool_outcomes(success) is None
