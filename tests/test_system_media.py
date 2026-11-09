"""
Automated unit tests for SystemMediaTool.

All Windows API calls are mocked; no real machine media keys, audio settings,
or active playback sessions are modified during test execution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch
import pytest

from core.tool_router import ToolCall, ToolRouter
from core.tool_setup import create_tool_registry
from core.intent_router import IntentRouter
from tools.base import ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry
from tools.system_media import (
    ACTION_MESSAGES,
    SystemMediaTool,
    VK_MEDIA_NEXT_TRACK,
    VK_MEDIA_PLAY_PAUSE,
    VK_MEDIA_PREV_TRACK,
    register_system_media_tools,
)


# ==============================================================================
# 1. Tool-Level Tests (Mocked Windows API)
# ==============================================================================

@pytest.mark.parametrize(
    "action,expected_vk,expected_msg",
    [
        ("play_pause", VK_MEDIA_PLAY_PAUSE, ACTION_MESSAGES["play_pause"]),
        ("next_track", VK_MEDIA_NEXT_TRACK, ACTION_MESSAGES["next_track"]),
        ("previous_track", VK_MEDIA_PREV_TRACK, ACTION_MESSAGES["previous_track"]),
    ],
)
@patch("platform.system", return_value="Windows")
def test_system_media_dispatches_correct_virtual_key(mock_platform, action, expected_vk, expected_msg):
    mock_sender = MagicMock()
    tool = SystemMediaTool(key_sender=mock_sender)

    result = tool.run(action=action)

    assert result.success is True
    assert result.error is None
    assert result.data["action"] == action
    assert result.data["vk_code"] == expected_vk
    assert result.message == expected_msg
    mock_sender.assert_called_once_with(expected_vk)


@patch("platform.system", return_value="Windows")
def test_system_media_rejects_unsupported_action(mock_platform):
    mock_sender = MagicMock()
    tool = SystemMediaTool(key_sender=mock_sender)

    result = tool.run(action="fast_forward")

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"
    assert "not supported" in result.message
    mock_sender.assert_not_called()


@patch("platform.system", return_value="Windows")
def test_system_media_rejects_empty_action(mock_platform):
    mock_sender = MagicMock()
    tool = SystemMediaTool(key_sender=mock_sender)

    result = tool.run(action="")

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"
    mock_sender.assert_not_called()


def test_registry_schema_validation_blocks_invalid_media_action():
    mock_sender = MagicMock()
    registry = ToolRegistry()
    registry.register(SystemMediaTool(key_sender=mock_sender))

    result = registry.execute("system_media", action="unsupported_cmd")

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"
    mock_sender.assert_not_called()


@patch("platform.system", return_value="Linux")
def test_system_media_returns_failure_on_non_windows(mock_platform):
    mock_sender = MagicMock()
    tool = SystemMediaTool(key_sender=mock_sender)

    result = tool.run(action="play_pause")

    assert result.success is False
    assert result.error_code == "UNSUPPORTED_PLATFORM"
    assert "only supported on Windows" in result.message
    mock_sender.assert_not_called()


@patch("platform.system", return_value="Darwin")
def test_system_media_returns_failure_on_macos(mock_platform):
    mock_sender = MagicMock()
    tool = SystemMediaTool(key_sender=mock_sender)

    result = tool.run(action="next_track")

    assert result.success is False
    assert result.error_code == "UNSUPPORTED_PLATFORM"
    mock_sender.assert_not_called()


@patch("platform.system", return_value="Windows")
def test_system_media_catches_key_sender_exception_safely(mock_platform):
    def failing_sender(vk_code):
        raise OSError("Simulated Windows user32 key dispatch error")

    tool = SystemMediaTool(key_sender=failing_sender)

    result = tool.run(action="play_pause")

    assert result.success is False
    assert result.error_code == "EXECUTION_ERROR"
    assert "Failed to execute media action" in result.message


def test_system_media_metadata_and_schema():
    tool = SystemMediaTool()
    assert tool.name == "system_media"
    assert tool.permission_level == PermissionLevel.SAFE
    assert isinstance(tool.description, str) and tool.description.strip()

    schema = tool.get_schema()
    assert schema["name"] == "system_media"
    assert schema["permission_level"] == "SAFE"
    assert "action" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["action"]
    assert set(schema["parameters"]["properties"]["action"]["enum"]) == {
        "play_pause", "next_track", "previous_track"
    }


# ==============================================================================
# 2. Registration Tests
# ==============================================================================

def test_system_media_registers_in_catalog():
    registry = create_tool_registry()
    tool = registry.get("system_media")

    assert tool is not None
    assert isinstance(tool, SystemMediaTool)
    assert tool.permission_level == PermissionLevel.SAFE
    assert "system_media" in registry.list_names()
    # Confirm existing system audio tool is also present
    assert "system_audio" in registry.list_names()


def test_register_system_media_tools_is_idempotent():
    registry = ToolRegistry()
    register_system_media_tools(registry)
    register_system_media_tools(registry)

    assert registry.has("system_media")
    assert len([name for name in registry.list_names() if name == "system_media"]) == 1


# ==============================================================================
# 3. ToolRouter and Permission Tests
# ==============================================================================

@patch("platform.system", return_value="Windows")
def test_tool_router_executes_system_media_without_confirmation(mock_platform):
    mock_sender = MagicMock()
    registry = ToolRegistry()
    registry.register(SystemMediaTool(key_sender=mock_sender))

    confirmation_mock = MagicMock()
    router = ToolRouter(registry, confirmation_handler=confirmation_mock)

    call = ToolCall(tool="system_media", arguments={"action": "play_pause"})
    result = router.execute_tool_call(call)

    assert result.success is True
    assert result.message == ACTION_MESSAGES["play_pause"]
    mock_sender.assert_called_once_with(VK_MEDIA_PLAY_PAUSE)
    confirmation_mock.request_confirmation.assert_not_called()


@patch("platform.system", return_value="Windows")
def test_tool_router_schema_validation_rejects_invalid_media_before_dispatch(mock_platform):
    mock_sender = MagicMock()
    registry = ToolRegistry()
    registry.register(SystemMediaTool(key_sender=mock_sender))

    confirmation_mock = MagicMock()
    router = ToolRouter(registry, confirmation_handler=confirmation_mock)

    call = ToolCall(tool="system_media", arguments={"action": "invalid_verb", "extra": 123})
    result = router.execute_tool_call(call)

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"
    mock_sender.assert_not_called()
    confirmation_mock.request_confirmation.assert_not_called()


# ==============================================================================
# 4. Intent Routing Tests
# ==============================================================================

@pytest.mark.parametrize(
    "phrase,expected_action",
    [
        ("Pause the music.", "play_pause"),
        ("Pause playback.", "play_pause"),
        ("Resume playback.", "play_pause"),
        ("Resume my music.", "play_pause"),
        ("Toggle media playback.", "play_pause"),
        ("Play or pause my music.", "play_pause"),
        ("pause", "play_pause"),
        ("resume", "play_pause"),
        ("Next track.", "next_track"),
        ("Next song.", "next_track"),
        ("Skip this track.", "next_track"),
        ("Skip this song.", "next_track"),
        ("Play the next song.", "next_track"),
        ("Previous track.", "previous_track"),
        ("Previous song.", "previous_track"),
        ("Go back one track.", "previous_track"),
        ("Go back a track.", "previous_track"),
        ("Play the previous song.", "previous_track"),
    ],
)
def test_intent_router_recognizes_clear_media_actions(phrase, expected_action):
    router = IntentRouter()
    route = router.route(phrase)

    assert route is not None
    assert route.intent == "SYSTEM_MEDIA"
    assert len(route.steps) == 1
    assert route.steps[0].intent == "SYSTEM_MEDIA"
    assert route.steps[0].call.tool == "system_media"
    assert route.steps[0].call.arguments == {"action": expected_action}


@pytest.mark.parametrize(
    "phrase",
    [
        "How does media playback work?",
        "Explain Windows media keys.",
        "What is the difference between play and pause?",
        "Play a video in VLC.",
        "Play a game.",
        "Play chess.",
        "Play with my dog.",
        "Open VLC and play a video.",
        "next song; rm -rf /",
        "powershell next track",
    ],
)
def test_intent_router_safeguards_against_misrouting_and_injection(phrase):
    router = IntentRouter()
    route = router.route(phrase)

    if route is not None:
        assert route.intent != "SYSTEM_MEDIA"


def test_existing_open_application_intents_unaffected_by_media_rules():
    router = IntentRouter()

    route_notepad = router.route("Open Notepad.")
    assert route_notepad is not None
    assert route_notepad.intent == "OPEN_APPLICATION"
    assert route_notepad.steps[0].call.arguments == {"app_name": "Notepad"}

    route_calc = router.route("Launch Calculator.")
    assert route_calc is not None
    assert route_calc.intent == "OPEN_APPLICATION"
    assert route_calc.steps[0].call.arguments == {"app_name": "Calculator"}
