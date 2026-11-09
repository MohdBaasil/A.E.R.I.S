from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai.provider import AIProvider
from core.assistant import Assistant
from core.confirmation import AutoApproveHandler, AutoDenyHandler
from core.intent_router import IntentRouter
from core.memory import MemoryStore
from core.tool_router import ToolRouter
from tools.applications import OpenApplicationTool
from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry
from tools.system import SystemInfoTool


class CaptureProvider(AIProvider):
    def __init__(self, replies=None):
        self.replies = list(replies or ["ordinary response"])
        self.requests = []

    def generate(self, messages):
        self.requests.append(list(messages))
        return self.replies.pop(0) if self.replies else "ordinary response"


@pytest.mark.parametrize("phrase", [
    "what is my cpu usage",
    "check my cpu",
    "how much ram am I using",
    "check my RAM",
    "show my disk usage",
    "how much battery do I have",
    "check my system",
    "show system information",
    "what is cpu usage?",
])
def test_current_system_requests_route_to_registered_system_info_intent(phrase):
    route = IntentRouter().route(phrase)
    assert route is not None
    assert route.intent == "SYSTEM_INFO"
    assert route.steps[0].call.tool == "system_info"
    assert route.steps[0].call.arguments == {}


@pytest.mark.parametrize("phrase", [
    "What is CPU?",
    "How does RAM work?",
    "Explain disk usage",
    "What does battery health mean?",
    "check it",
    "list files",
])
def test_educational_and_ambiguous_requests_are_not_routed(phrase):
    assert IntentRouter().route(phrase) is None


@pytest.mark.parametrize(("phrase", "app_name"), [
    ("open VS Code", "VS Code"),
    ("launch Notepad", "Notepad"),
    ("start Chrome", "Chrome"),
])
def test_explicit_application_intents_route_without_launching(phrase, app_name):
    route = IntentRouter().route(phrase)
    assert route is not None
    assert route.intent == "OPEN_APPLICATION"
    assert route.steps[0].call.tool == "open_application"
    assert route.steps[0].call.arguments == {"app_name": app_name}


@pytest.mark.parametrize("phrase", [
    "run powershell Get-Process",
    "execute cmd /c whoami",
    "run this command to open Notepad",
    "run powershell",
])
def test_shell_and_command_requests_are_not_converted_to_application_launches(phrase):
    assert IntentRouter().route(phrase) is None


@pytest.mark.parametrize("phrase", [
    "show the files here",
    "list files in this folder",
    "what files are in this directory",
    "show my project files",
])
def test_clear_file_listing_aliases_use_known_current_directory(phrase, monkeypatch):
    monkeypatch.setattr("core.intent_router.os.getcwd", lambda: r"C:\AERIS\workspace")
    route = IntentRouter().route(phrase)
    assert route is not None
    assert route.intent == "LIST_FILES"
    assert route.steps[0].call.tool == "list_files"
    assert route.steps[0].call.arguments == {"path": r"C:\AERIS\workspace"}


def test_explicit_path_is_preserved_and_bare_ambiguous_listing_is_not_routed():
    explicit = IntentRouter().route(r"list files in D:\Projects\AERIS")
    assert explicit is not None
    assert explicit.steps[0].call.arguments["path"] == r"D:\Projects\AERIS"
    assert IntentRouter().route("show files") is None


def test_memory_commands_remain_owned_by_memory_layer():
    assert IntentRouter().route("Remember that I use VS Code.") is None


class FakeOpenApplication(Tool):
    name = "open_application"
    description = "Open a registered application."
    parameters = {
        "type": "object",
        "properties": {"app_name": {"type": "string"}},
        "required": ["app_name"],
    }
    permission_level = PermissionLevel.SAFE

    def __init__(self, events, result=None):
        super().__init__()
        self.events = events
        self.result = result or ToolResult.ok("Opened Notepad.", {"display_name": "Notepad"})

    def execute(self, app_name="", **kwargs):
        self.events.append(("open_application", app_name))
        return self.result


class FakeListFiles(Tool):
    name = "list_files"
    description = "List a directory."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    permission_level = PermissionLevel.SAFE

    def __init__(self, events, result=None):
        super().__init__()
        self.events = events
        self.result = result or ToolResult.ok(
            "Listed 2 item(s).",
            {"path": ".", "total_listed": 2, "items": [
                {"name": "README.md", "type": "file"},
                {"name": "src", "type": "directory"},
            ]},
        )

    def execute(self, path="", **kwargs):
        self.events.append(("list_files", path))
        return self.result


def _assistant_for_multi(events, *, open_result=None, list_result=None, provider=None):
    registry = ToolRegistry()
    registry.register(FakeOpenApplication(events, open_result))
    registry.register(FakeListFiles(events, list_result))
    provider = provider or CaptureProvider()
    assistant = Assistant(provider, ToolRouter(registry, AutoApproveHandler()))
    return assistant, provider


def test_multi_action_executes_in_order_and_returns_actual_results(monkeypatch):
    monkeypatch.setattr("core.intent_router.os.getcwd", lambda: r"C:\workspace")
    events = []
    assistant, provider = _assistant_for_multi(events)

    reply = assistant.send("Open Notepad and then list the files.")

    assert [event[0] for event in events] == ["open_application", "list_files"]
    assert events[0][1] == "Notepad"
    assert events[1][1] == r"C:\workspace"
    assert "Opened Notepad." in reply
    assert "README.md" in reply and "src" in reply
    assert provider.requests == []
    assert sum("```tool_result" in message["content"] for message in assistant.history) == 2


def test_first_step_failure_stops_dependent_file_listing(monkeypatch):
    monkeypatch.setattr("core.intent_router.os.getcwd", lambda: r"C:\workspace")
    events = []
    assistant, _ = _assistant_for_multi(
        events,
        open_result=ToolResult.fail("NOT_FOUND", "Notepad executable was not found."),
    )

    reply = assistant.send("Open Notepad and then list the files.")

    assert [event[0] for event in events] == ["open_application"]
    assert "operation failed" in reply
    assert "README.md" not in reply


def test_second_step_failure_keeps_first_success_and_reports_failure(monkeypatch):
    monkeypatch.setattr("core.intent_router.os.getcwd", lambda: r"C:\workspace")
    events = []
    assistant, _ = _assistant_for_multi(
        events,
        list_result=ToolResult.fail("PATH_NOT_FOUND", "Requested directory was not found."),
    )

    reply = assistant.send("Open Notepad and then list the files.")

    assert [event[0] for event in events] == ["open_application", "list_files"]
    assert "Opened Notepad." in reply
    assert "list files operation failed" in reply


def test_deterministic_system_info_uses_tool_result_without_provider_claims():
    provider = CaptureProvider(["CPU usage is 99%."])
    registry = ToolRegistry()
    system_info = SystemInfoTool()
    system_info.execute = MagicMock(return_value=ToolResult.ok(
        "System metrics collected.",
        {
            "cpu": {"available": True, "usage_percent": 17.4},
            "memory": {"available": True, "used_gb": 4.0, "total_gb": 8.0, "usage_percent": 50.0},
            "disk": {"available": True, "used_gb": 20.0, "total_gb": 100.0, "usage_percent": 20.0},
            "battery": {"available": False},
            "os": {"available": True, "system": "Windows", "release": "11", "version": "11", "machine": "x64"},
        },
    ))
    registry.register(system_info)
    assistant = Assistant(provider, ToolRouter(registry))

    reply = assistant.send("what is my cpu usage?")

    assert reply == "CPU usage is currently 17.4%."
    system_info.execute.assert_called_once_with()


def test_allowed_application_uses_tool_allowlist_and_reports_tool_result():
    registry = ToolRegistry()
    tool = OpenApplicationTool()
    tool._find_executable = MagicMock(return_value="notepad.exe")
    registry.register(tool)
    provider = CaptureProvider()
    assistant = Assistant(provider, ToolRouter(registry))

    with patch("tools.applications.subprocess.Popen") as popen:
        popen.return_value.pid = 4321
        reply = assistant.send("open Notepad")

    assert reply == "Opened Notepad."
    popen.assert_called_once_with(["notepad.exe"])


def test_unsupported_application_is_rejected_by_registered_tool():
    registry = ToolRegistry()
    registry.register(OpenApplicationTool())
    assistant = Assistant(CaptureProvider(), ToolRouter(registry))

    reply = assistant.send("open an unsupported test application")

    assert "not supported by AERIS" in reply
    assert "was not opened" in reply


@pytest.mark.parametrize("permission", [PermissionLevel.CONFIRM, PermissionLevel.DANGEROUS])
def test_deterministic_route_cannot_bypass_confirmation(permission):
    events = []

    class GatedOpenTool(FakeOpenApplication):
        pass

    GatedOpenTool.permission_level = permission

    registry = ToolRegistry()
    tool = GatedOpenTool(events)
    registry.register(tool)
    confirmation = MagicMock()
    confirmation.request_confirmation.return_value = False
    assistant = Assistant(CaptureProvider(), ToolRouter(registry, confirmation))

    reply = assistant.send("open Notepad")

    assert events == []
    confirmation.request_confirmation.assert_called_once()
    assert permission.value in confirmation.request_confirmation.call_args.kwargs["permission_level"]
    assert "cancelled" in reply


def test_deterministic_route_cannot_bypass_schema_validation():
    events = []

    class InvalidSchemaOpenTool(FakeOpenApplication):
        parameters = {
            "type": "object",
            "properties": {"app_name": {"type": "integer"}},
            "required": ["app_name"],
        }

    registry = ToolRegistry()
    registry.register(InvalidSchemaOpenTool(events))
    confirmation = MagicMock()
    assistant = Assistant(CaptureProvider(), ToolRouter(registry, confirmation))

    reply = assistant.send("open Notepad")

    assert events == []
    confirmation.request_confirmation.assert_not_called()
    assert "failed" in reply


def test_deterministic_intents_work_in_streaming_and_respect_cancellation():
    events = []
    assistant, provider = _assistant_for_multi(events)
    chunks = list(assistant.send_stream("Open Notepad"))
    assert chunks == ["Opened Notepad."]
    assert "tool_call" not in "".join(chunks)
    # No LLM follow-up call expected for deterministic OPEN_APPLICATION intent
    assert provider.requests == []

    import threading
    cancelled = threading.Event()
    cancelled.set()
    second_events = []
    second, _ = _assistant_for_multi(second_events)
    assert list(second.send_stream("Open Notepad", cancel_event=cancelled)) == []
    assert second_events == []


def test_stream_cancellation_between_steps_prevents_dependent_execution(monkeypatch):
    import threading

    monkeypatch.setattr("core.intent_router.os.getcwd", lambda: r"C:\workspace")
    cancel_event = threading.Event()
    events = []

    class CancellingOpenTool(FakeOpenApplication):
        def execute(self, app_name="", **kwargs):
            result = super().execute(app_name, **kwargs)
            cancel_event.set()
            return result

    registry = ToolRegistry()
    registry.register(CancellingOpenTool(events))
    registry.register(FakeListFiles(events))
    assistant = Assistant(CaptureProvider(), ToolRouter(registry, AutoApproveHandler()))

    chunks = list(assistant.send_stream("Open Notepad and then list the files.", cancel_event=cancel_event))

    assert [event[0] for event in events] == ["open_application"]
    assert len(chunks) == 1
    assert "Opened Notepad." in chunks[0]
    assert "cancelled" in chunks[0]


def test_router_rejects_missing_deterministic_tool_registration():
    provider = CaptureProvider()
    assistant = Assistant(provider, ToolRouter(ToolRegistry()))
    reply = assistant.send("what is my cpu usage?")
    assert "tool is unavailable" in reply
    # No LLM follow-up call: SYSTEM_INFO is deterministic even when the tool is missing
    assert provider.requests == []


def test_memory_commands_remain_deterministic_before_intent_routing(tmp_path):
    provider = CaptureProvider()
    assistant = Assistant(provider, memory_store=MemoryStore(tmp_path / "memory.sqlite3"))
    assert "remember" in assistant.send("Remember that I use VS Code.").lower()
    assert provider.requests == []
