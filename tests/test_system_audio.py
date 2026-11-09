"""
Automated unit tests for SystemAudioTool.

All Windows API calls are mocked; no real machine audio settings or key events
are modified during test execution.
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
from tools.system_audio import (
    SystemAudioTool,
    VK_VOLUME_DOWN,
    VK_VOLUME_UP,
    register_system_audio_tools,
)


# ==============================================================================
# 1. Volume Step Dispatch Tests (Mocked Windows API)
# ==============================================================================

@pytest.mark.parametrize(
    "action,expected_vk,expected_msg",
    [
        ("volume_up", VK_VOLUME_UP, "Increased system volume."),
        ("volume_down", VK_VOLUME_DOWN, "Decreased system volume."),
    ],
)
@patch("platform.system", return_value="Windows")
def test_system_audio_dispatches_volume_virtual_keys(mock_platform, action, expected_vk, expected_msg):
    mock_sender = MagicMock()
    mock_getter = MagicMock()
    mock_setter = MagicMock()
    tool = SystemAudioTool(key_sender=mock_sender, mute_getter=mock_getter, mute_setter=mock_setter)

    result = tool.run(action=action)

    assert result.success is True
    assert result.error is None
    assert result.data["action"] == action
    assert result.data["vk_code"] == expected_vk
    assert result.message == expected_msg
    mock_sender.assert_called_once_with(expected_vk)
    mock_getter.assert_not_called()
    mock_setter.assert_not_called()


# ==============================================================================
# 2. Idempotent Mute and Unmute Tests
# ==============================================================================

@patch("platform.system", return_value="Windows")
def test_mute_when_unmuted_sets_mute_to_true(mock_platform):
    mock_sender = MagicMock()
    mock_getter = MagicMock(return_value=False)
    mock_setter = MagicMock()
    tool = SystemAudioTool(key_sender=mock_sender, mute_getter=mock_getter, mute_setter=mock_setter)

    result = tool.run(action="mute")

    assert result.success is True
    assert result.data == {"action": "mute", "muted": True, "changed": True}
    assert result.message == "Muted system audio."
    mock_getter.assert_called_once()
    mock_setter.assert_called_once_with(True)
    mock_sender.assert_not_called()


@patch("platform.system", return_value="Windows")
def test_repeated_mute_when_already_muted_is_idempotent_no_op(mock_platform):
    mock_sender = MagicMock()
    mock_getter = MagicMock(return_value=True)  # Already muted
    mock_setter = MagicMock()
    tool = SystemAudioTool(key_sender=mock_sender, mute_getter=mock_getter, mute_setter=mock_setter)

    result = tool.run(action="mute")

    assert result.success is True
    assert result.data == {"action": "mute", "muted": True, "changed": False}
    assert result.message == "System audio is already muted."
    mock_getter.assert_called_once()
    # Critical idempotency guarantee: do not toggle or re-apply
    mock_setter.assert_not_called()
    mock_sender.assert_not_called()


@patch("platform.system", return_value="Windows")
def test_unmute_when_muted_sets_mute_to_false(mock_platform):
    mock_sender = MagicMock()
    mock_getter = MagicMock(return_value=True)  # Currently muted
    mock_setter = MagicMock()
    tool = SystemAudioTool(key_sender=mock_sender, mute_getter=mock_getter, mute_setter=mock_setter)

    result = tool.run(action="unmute")

    assert result.success is True
    assert result.data == {"action": "unmute", "muted": False, "changed": True}
    assert result.message == "Unmuted system audio."
    mock_getter.assert_called_once()
    mock_setter.assert_called_once_with(False)
    mock_sender.assert_not_called()


@patch("platform.system", return_value="Windows")
def test_repeated_unmute_when_already_unmuted_is_idempotent_no_op(mock_platform):
    mock_sender = MagicMock()
    mock_getter = MagicMock(return_value=False)  # Already unmuted
    mock_setter = MagicMock()
    tool = SystemAudioTool(key_sender=mock_sender, mute_getter=mock_getter, mute_setter=mock_setter)

    result = tool.run(action="unmute")

    assert result.success is True
    assert result.data == {"action": "unmute", "muted": False, "changed": False}
    assert result.message == "System audio is already unmuted."
    mock_getter.assert_called_once()
    mock_setter.assert_not_called()
    mock_sender.assert_not_called()


@patch("platform.system", return_value="Windows")
def test_stateful_mute_and_unmute_sequence(mock_platform):
    """Simulate sequential calls tracking state changes."""
    state = {"muted": False}

    def fake_getter():
        return state["muted"]

    def fake_setter(mute_val):
        state["muted"] = mute_val

    tool = SystemAudioTool(key_sender=MagicMock(), mute_getter=fake_getter, mute_setter=fake_setter)

    # 1. First mute: changes from unmuted to muted
    r1 = tool.run(action="mute")
    assert r1.success is True
    assert r1.data["changed"] is True
    assert state["muted"] is True

    # 2. Second mute: already muted, stays muted! (No toggle defect)
    r2 = tool.run(action="mute")
    assert r2.success is True
    assert r2.data["changed"] is False
    assert state["muted"] is True

    # 3. Third mute: still stays muted!
    r3 = tool.run(action="mute")
    assert r3.success is True
    assert r3.data["changed"] is False
    assert state["muted"] is True

    # 4. First unmute: changes from muted to unmuted
    r4 = tool.run(action="unmute")
    assert r4.success is True
    assert r4.data["changed"] is True
    assert state["muted"] is False

    # 5. Second unmute: already unmuted, stays unmuted!
    r5 = tool.run(action="unmute")
    assert r5.success is True
    assert r5.data["changed"] is False
    assert state["muted"] is False


# ==============================================================================
# 3. Argument Validation & Unknown Action Tests
# ==============================================================================

@patch("platform.system", return_value="Windows")
def test_system_audio_rejects_unsupported_action(mock_platform):
    mock_sender = MagicMock()
    tool = SystemAudioTool(key_sender=mock_sender, mute_getter=MagicMock(), mute_setter=MagicMock())

    result = tool.run(action="set_to_100")

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"
    assert "not supported" in result.message
    mock_sender.assert_not_called()


@patch("platform.system", return_value="Windows")
def test_system_audio_rejects_empty_action(mock_platform):
    mock_sender = MagicMock()
    tool = SystemAudioTool(key_sender=mock_sender, mute_getter=MagicMock(), mute_setter=MagicMock())

    result = tool.run(action="")

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"
    mock_sender.assert_not_called()


def test_registry_schema_validation_blocks_invalid_action():
    mock_sender = MagicMock()
    registry = ToolRegistry()
    registry.register(SystemAudioTool(key_sender=mock_sender, mute_getter=MagicMock(), mute_setter=MagicMock()))

    result = registry.execute("system_audio", action="invalid_action")

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"
    mock_sender.assert_not_called()


# ==============================================================================
# 4. Non-Windows System Compatibility
# ==============================================================================

@patch("platform.system", return_value="Linux")
def test_system_audio_returns_failure_on_non_windows(mock_platform):
    mock_sender = MagicMock()
    mock_getter = MagicMock()
    mock_setter = MagicMock()
    tool = SystemAudioTool(key_sender=mock_sender, mute_getter=mock_getter, mute_setter=mock_setter)

    result = tool.run(action="volume_up")

    assert result.success is False
    assert result.error_code == "UNSUPPORTED_PLATFORM"
    assert "only supported on Windows" in result.message
    mock_sender.assert_not_called()
    mock_getter.assert_not_called()
    mock_setter.assert_not_called()


@patch("platform.system", return_value="Darwin")
def test_system_audio_returns_failure_on_macos(mock_platform):
    mock_sender = MagicMock()
    mock_getter = MagicMock()
    mock_setter = MagicMock()
    tool = SystemAudioTool(key_sender=mock_sender, mute_getter=mock_getter, mute_setter=mock_setter)

    result = tool.run(action="mute")

    assert result.success is False
    assert result.error_code == "UNSUPPORTED_PLATFORM"
    mock_sender.assert_not_called()
    mock_getter.assert_not_called()
    mock_setter.assert_not_called()


# ==============================================================================
# 5. Exception Handling
# ==============================================================================

@patch("platform.system", return_value="Windows")
def test_system_audio_catches_key_sender_exception_safely(mock_platform):
    def failing_sender(vk_code):
        raise OSError("Simulated Windows user32 dispatch error")

    tool = SystemAudioTool(key_sender=failing_sender)

    result = tool.run(action="volume_up")

    assert result.success is False
    assert result.error_code == "EXECUTION_ERROR"
    assert "Failed to execute audio action" in result.message


@patch("platform.system", return_value="Windows")
def test_system_audio_catches_mute_getter_exception_safely(mock_platform):
    def failing_getter():
        raise OSError("Simulated Core Audio endpoint query error")

    tool = SystemAudioTool(mute_getter=failing_getter, mute_setter=MagicMock())

    result = tool.run(action="mute")

    assert result.success is False
    assert result.error_code == "EXECUTION_ERROR"
    assert "Failed to execute audio action" in result.message


@patch("platform.system", return_value="Windows")
def test_system_audio_catches_mute_setter_exception_safely(mock_platform):
    def failing_setter(mute_val):
        raise OSError("Simulated Core Audio endpoint set error")

    tool = SystemAudioTool(mute_getter=MagicMock(return_value=False), mute_setter=failing_setter)

    result = tool.run(action="mute")

    assert result.success is False
    assert result.error_code == "EXECUTION_ERROR"
    assert "Failed to execute audio action" in result.message


# ==============================================================================
# 6. Registry Registration & Tool Metadata
# ==============================================================================

def test_system_audio_metadata_and_schema():
    tool = SystemAudioTool()
    assert tool.name == "system_audio"
    assert tool.permission_level == PermissionLevel.SAFE
    assert isinstance(tool.description, str) and tool.description.strip()

    schema = tool.get_schema()
    assert schema["name"] == "system_audio"
    assert schema["permission_level"] == "SAFE"
    assert "action" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["action"]
    assert set(schema["parameters"]["properties"]["action"]["enum"]) == {
        "volume_up", "volume_down", "mute", "unmute"
    }


def test_system_audio_registers_in_catalog():
    registry = create_tool_registry()
    tool = registry.get("system_audio")

    assert tool is not None
    assert isinstance(tool, SystemAudioTool)
    assert tool.permission_level == PermissionLevel.SAFE
    assert "system_audio" in registry.list_names()


# ==============================================================================
# 7. ToolRouter Permission & Execution Flow
# ==============================================================================

@patch("platform.system", return_value="Windows")
def test_tool_router_executes_system_audio_without_confirmation(mock_platform):
    mock_sender = MagicMock()
    registry = ToolRegistry()
    registry.register(SystemAudioTool(key_sender=mock_sender, mute_getter=MagicMock(), mute_setter=MagicMock()))

    confirmation_mock = MagicMock()
    router = ToolRouter(registry, confirmation_handler=confirmation_mock)

    call = ToolCall(tool="system_audio", arguments={"action": "volume_up"})
    result = router.execute_tool_call(call)

    assert result.success is True
    assert result.message == "Increased system volume."
    mock_sender.assert_called_once_with(VK_VOLUME_UP)
    # PermissionLevel is SAFE: confirmation should NOT be requested
    confirmation_mock.request_confirmation.assert_not_called()


# ==============================================================================
# 8. Intent Router Integration Tests
# ==============================================================================

@pytest.mark.parametrize(
    "phrase,expected_action",
    [
        ("volume up", "volume_up"),
        ("Volume Up.", "volume_up"),
        ("Please turn up the volume", "volume_up"),
        ("increase the volume", "volume_up"),
        ("make it louder", "volume_up"),
        ("louder", "volume_up"),
        ("volume down", "volume_down"),
        ("turn down the volume", "volume_down"),
        ("decrease volume", "volume_down"),
        ("lower the volume", "volume_down"),
        ("make it quieter", "volume_down"),
        ("quieter", "volume_down"),
        ("softer", "volume_down"),
        ("mute", "mute"),
        ("mute audio", "mute"),
        ("mute the sound", "mute"),
        ("turn off sound", "mute"),
        ("unmute", "unmute"),
        ("unmute audio", "unmute"),
        ("turn on the sound", "unmute"),
    ],
)
def test_intent_router_recognizes_audio_actions(phrase, expected_action):
    router = IntentRouter()
    route = router.route(phrase)

    assert route is not None
    assert route.intent == "SYSTEM_AUDIO"
    assert len(route.steps) == 1
    assert route.steps[0].intent == "SYSTEM_AUDIO"
    assert route.steps[0].call.tool == "system_audio"
    assert route.steps[0].call.arguments == {"action": expected_action}


@pytest.mark.parametrize(
    "phrase",
    [
        "How does volume control work?",
        "Explain mute",
        "What is volume?",
        "volume up; rm -rf /",
        "powershell mute",
    ],
)
def test_intent_router_ignores_educational_and_command_injection(phrase):
    router = IntentRouter()
    route = router.route(phrase)

    if route is not None:
        assert route.intent != "SYSTEM_AUDIO"
