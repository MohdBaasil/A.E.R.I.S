"""Deterministic pre-routing for explicit, current system-information queries."""

from unittest.mock import MagicMock

import pytest

from ai.provider import AIProvider
from core.assistant import Assistant, _is_live_system_info_request
from core.tool_router import ToolRouter
from core.tool_setup import create_tool_registry
from tools.base import ToolResult
from tools.permissions import PermissionLevel


class CapturingProvider(AIProvider):
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def generate(self, messages):
        self.requests.append([dict(message) for message in messages])
        return self.replies.pop(0)


def _system_metrics(cpu=42.3, memory=67.8, disk=54.2, battery=73):
    return {
        "cpu": {"available": True, "usage_percent": cpu},
        "memory": {"available": True, "used_gb": 10.0, "total_gb": 16.0, "usage_percent": memory},
        "disk": {"available": True, "path": "C:\\", "used_gb": 100.0, "total_gb": 200.0, "usage_percent": disk},
        "battery": {"available": True, "percent": battery, "plugged_in": False},
        "os": {"available": True, "system": "Windows", "release": "11", "version": "test", "machine": "AMD64"},
    }


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is my CPU usage?", "CPU usage is currently 42.3%."),
        ("Check my RAM usage.", "RAM usage is currently 67.8% (10.0 GB used of 16.0 GB)."),
        ("What is my disk usage?", "Disk usage is currently 54.2% (100.0 GB used of 200.0 GB)."),
        ("What is my battery status?", "Battery: 73%, on battery."),
        ("What is my system information?", None),
        ("Tell me about my battery", "Battery: 73%, on battery."),
    ],
)
def test_live_system_queries_execute_registered_tool_and_use_its_result(query, expected):
    registry = create_tool_registry()
    system_tool = registry.get("system_info")
    system_tool.execute = MagicMock(
        return_value=ToolResult.ok("live metrics", data=_system_metrics())
    )
    provider = CapturingProvider(["Model response with potentially invented details."])
    confirmation = MagicMock()
    assistant = Assistant(provider, ToolRouter(registry, confirmation_handler=confirmation))

    answer = assistant.send(query)

    assert system_tool.execute.call_count == 1
    if expected is None:
        assert "CPU usage is currently 42.3%." in answer
        assert "RAM usage is currently 67.8%" in answer
        assert "Battery: 73%, on battery." in answer
    else:
        assert answer == expected
    confirmation.request_confirmation.assert_not_called()
    dialog = assistant.history[2:]
    assert dialog[0] == {"role": "user", "content": query}
    assert dialog[1]["role"] == "assistant" and "```tool_call" in dialog[1]["content"]
    assert dialog[2]["role"] == "user" and "```tool_result" in dialog[2]["content"]
    assert dialog[3] == {"role": "assistant", "content": answer}


@pytest.mark.parametrize(
    "query",
    [
        "How does CPU utilization work?",
        "What is a CPU?",
        "What is RAM?",
        "Explain RAM.",
        "Tell me about battery technology.",
    ],
)
def test_conceptual_system_questions_do_not_execute_system_info(query):
    registry = create_tool_registry()
    system_tool = registry.get("system_info")
    system_tool.execute = MagicMock(return_value=ToolResult.ok("should not run"))
    provider = CapturingProvider(["Here is a general explanation."])
    assistant = Assistant(provider, ToolRouter(registry))

    assert assistant.send(query) == "Here is a general explanation."
    system_tool.execute.assert_not_called()


@pytest.mark.parametrize(
    ("first_query", "second_query", "second_cpu", "second_memory"),
    [
        ("What is my CPU usage?", "What is my RAM usage?", 51.0, 61.0),
        ("Check my RAM usage", "What is my system information?", 52.0, 62.0),
    ],
)
def test_consecutive_queries_get_fresh_system_info_results(
    first_query, second_query, second_cpu, second_memory
):
    registry = create_tool_registry()
    system_tool = registry.get("system_info")
    system_tool.execute = MagicMock(side_effect=[
        ToolResult.ok("first snapshot", data=_system_metrics(cpu=11.0, memory=21.0)),
        ToolResult.ok("second snapshot", data=_system_metrics(cpu=second_cpu, memory=second_memory)),
    ])
    provider = CapturingProvider(["first model reply", "second model reply"])
    assistant = Assistant(provider, ToolRouter(registry))

    assistant.send(first_query)
    second_answer = assistant.send(second_query)

    assert system_tool.execute.call_count == 2
    if "RAM" in second_query:
        assert f"{second_memory}%" in second_answer
    else:
        assert f"{second_cpu}%" in second_answer


def test_system_info_tool_failure_is_returned_without_model_success_claim():
    registry = create_tool_registry()
    system_tool = registry.get("system_info")
    system_tool.execute = MagicMock(
        return_value=ToolResult.fail("METRICS_UNAVAILABLE", "Unable to read metrics.")
    )
    provider = CapturingProvider(["CPU is at 99%."])
    assistant = Assistant(provider, ToolRouter(registry))

    answer = assistant.send("What is my CPU usage?")

    assert answer == "The system info operation failed, so it was not completed."
    assert system_tool.execute.call_count == 1


def test_streaming_live_query_is_pre_routed_and_uses_fresh_result():
    class StreamingProvider(CapturingProvider):
        def __init__(self):
            super().__init__([])
            self.stream_requests = []

        def generate_stream(self, messages, cancel_event=None):
            self.stream_requests.append([dict(message) for message in messages])
            yield "The CPU is 99% and RAM is 100%."

    registry = create_tool_registry()
    system_tool = registry.get("system_info")
    system_tool.execute = MagicMock(
        return_value=ToolResult.ok("fresh metrics", data=_system_metrics(cpu=18.7))
    )
    provider = StreamingProvider()
    assistant = Assistant(provider, ToolRouter(registry))

    chunks = list(assistant.send_stream("What is my CPU usage?"))

    assert "".join(chunks) == "CPU usage is currently 18.7%."
    system_tool.execute.assert_called_once_with()


def test_deterministic_pre_route_still_obeys_router_permission_gate():
    registry = create_tool_registry()
    system_tool = registry.get("system_info")
    system_tool.permission_level = PermissionLevel.CONFIRM
    system_tool.execute = MagicMock(return_value=ToolResult.ok("should not execute"))
    confirmation = MagicMock()
    confirmation.request_confirmation.return_value = False
    provider = CapturingProvider(["The CPU check succeeded."])
    assistant = Assistant(provider, ToolRouter(registry, confirmation_handler=confirmation))

    answer = assistant.send("What is my CPU usage?")

    assert "cancelled" in answer.lower()
    confirmation.request_confirmation.assert_called_once()
    system_tool.execute.assert_not_called()


def test_intent_classifier_uses_only_the_current_user_message():
    assert _is_live_system_info_request("What is my CPU usage?")
    assert not _is_live_system_info_request("What is a CPU?")
