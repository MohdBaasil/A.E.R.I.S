import subprocess
from unittest.mock import patch, MagicMock
import pytest

from tools.applications import OpenApplicationTool, register_application_tools
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry


# ==========================================
# Alias & Resolution Tests
# ==========================================

def test_alias_resolution_notepad():
    tool = OpenApplicationTool()
    for alias in ["notepad", "NOTEPAD", "  note  ", "Text Editor"]:
        resolved = tool.resolve_app(alias)
        assert resolved is not None
        assert resolved["app_key"] == "notepad"
        assert resolved["display_name"] == "Notepad"


def test_alias_resolution_calculator():
    tool = OpenApplicationTool()
    for alias in ["calculator", "CALC", "calc", "  calculator  "]:
        resolved = tool.resolve_app(alias)
        assert resolved is not None
        assert resolved["app_key"] == "calculator"
        assert resolved["display_name"] == "Calculator"


def test_alias_resolution_explorer():
    tool = OpenApplicationTool()
    for alias in ["explorer", "File Explorer", "files", "my computer"]:
        resolved = tool.resolve_app(alias)
        assert resolved is not None
        assert resolved["app_key"] == "explorer"
        assert resolved["display_name"] == "File Explorer"


def test_alias_resolution_chrome():
    tool = OpenApplicationTool()
    for alias in ["chrome", "Google Chrome", "google-chrome", "browser"]:
        resolved = tool.resolve_app(alias)
        assert resolved is not None
        assert resolved["app_key"] == "chrome"
        assert resolved["display_name"] == "Google Chrome"


def test_alias_resolution_vscode():
    tool = OpenApplicationTool()
    for alias in ["vscode", "VS Code", "code", "Visual Studio Code"]:
        resolved = tool.resolve_app(alias)
        assert resolved is not None
        assert resolved["app_key"] == "vscode"
        assert resolved["display_name"] == "Visual Studio Code"


def test_unknown_application_resolution():
    tool = OpenApplicationTool()
    for unknown in ["random_malware", "powershell", "cmd", "unknown_app", "calc.exe; rm -rf"]:
        assert tool.resolve_app(unknown) is None


# ==========================================
# Execution & Security Tests
# ==========================================

@patch("subprocess.Popen")
def test_open_application_success(mock_popen):
    mock_proc = MagicMock()
    mock_proc.pid = 1234
    mock_popen.return_value = mock_proc

    tool = OpenApplicationTool()
    result = tool.run(app_name="Notepad")

    assert result.success is True
    assert result.message == "Opened Notepad."
    assert result.data["app_key"] == "notepad"
    assert result.data["display_name"] == "Notepad"
    assert result.data["pid"] == 1234
    assert result.error is None

    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    # Verify no shell=True is passed for security
    assert kwargs.get("shell") is not True


@patch("subprocess.Popen")
def test_open_application_unknown_rejected(mock_popen):
    tool = OpenApplicationTool()
    result = tool.run(app_name="cmd.exe")

    assert result.success is False
    assert result.error == "UNKNOWN_APPLICATION"
    assert "not in the approved allowlist" in result.message
    # Critical security check: subprocess should never be invoked
    mock_popen.assert_not_called()


@patch("subprocess.Popen")
def test_open_application_empty_name_rejected(mock_popen):
    tool = OpenApplicationTool()
    result = tool.run(app_name="   ")

    assert result.success is False
    assert result.error == "INVALID_ARGUMENT"
    mock_popen.assert_not_called()


@patch("subprocess.Popen", side_effect=FileNotFoundError("Executable not found"))
def test_open_application_file_not_found(mock_popen):
    tool = OpenApplicationTool()
    result = tool.run(app_name="chrome")

    assert result.success is False
    assert result.error == "NOT_FOUND"
    assert "was not found on this system" in result.message


@patch("subprocess.Popen", side_effect=OSError("Access denied"))
def test_open_application_os_error(mock_popen):
    tool = OpenApplicationTool()
    result = tool.run(app_name="notepad")

    assert result.success is False
    assert result.error == "LAUNCH_FAILED"
    assert "Failed to launch 'Notepad'" in result.message


# ==========================================
# Metadata & Registry Tests
# ==========================================

def test_open_application_metadata():
    tool = OpenApplicationTool()
    assert tool.name == "open_application"
    assert tool.permission_level == PermissionLevel.SAFE
    assert "app_name" in tool.parameters["properties"]

    schema = tool.get_schema()
    assert schema["name"] == "open_application"
    assert schema["permission_level"] == "SAFE"


@patch("subprocess.Popen")
def test_registry_integration(mock_popen):
    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_popen.return_value = mock_proc

    registry = ToolRegistry()
    register_application_tools(registry)

    assert registry.has("open_application") is True
    tool = registry.get("open_application")
    assert isinstance(tool, OpenApplicationTool)

    result = registry.execute("open_application", app_name="calc")
    assert result.success is True
    assert result.data["display_name"] == "Calculator"


# ==========================================
# Brave Browser Allowlist Tests (Phase 10C)
# ==========================================

def test_alias_resolution_brave():
    tool = OpenApplicationTool()
    for alias in ["brave", "BRAVE", "  brave  ", "Brave Browser", "brave browser", "brave-browser"]:
        resolved = tool.resolve_app(alias)
        assert resolved is not None
        assert resolved["app_key"] == "brave"
        assert resolved["display_name"] == "Brave Browser"


@patch("os.path.isfile")
@patch("subprocess.Popen")
def test_open_application_brave_success_mocked(mock_popen, mock_isfile):
    mock_isfile.side_effect = lambda p: "brave.exe" in p.lower()
    mock_proc = MagicMock()
    mock_proc.pid = 4321
    mock_popen.return_value = mock_proc

    tool = OpenApplicationTool()
    result = tool.run(app_name="Brave Browser")

    assert result.success is True
    assert result.message == "Opened Brave Browser."
    assert result.data["app_key"] == "brave"
    assert result.data["display_name"] == "Brave Browser"
    assert result.data["pid"] == 4321

    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert kwargs.get("shell") is not True
    assert args[0][0].lower().endswith("brave.exe")


@patch("tools.applications.OpenApplicationTool._find_windows_app_path", return_value=None)
@patch("os.path.isfile", return_value=False)
@patch("subprocess.Popen")
def test_open_application_brave_not_found(mock_popen, mock_isfile, mock_app_path):
    tool = OpenApplicationTool()
    result = tool.run(app_name="brave")

    assert result.success is False
    assert result.error == "NOT_FOUND"
    assert "was not found in an approved location" in result.message
    mock_popen.assert_not_called()


def test_windows_app_path_rejects_non_basename():
    # Never accept path separators or directory traversals in executable target
    assert OpenApplicationTool._find_windows_app_path(r"..\..\evil.exe") is None
    assert OpenApplicationTool._find_windows_app_path(r"C:\evil.exe") is None
    assert OpenApplicationTool._find_windows_app_path("") is None


def test_brave_intent_routing():
    from core.intent_router import IntentRouter
    router = IntentRouter()

    for phrase, expected_arg in [
        ("open brave", "brave"),
        ("open brave browser", "brave browser"),
        ("launch brave browser", "brave browser"),
        ("start brave", "brave"),
    ]:
        route = router.route(phrase)
        assert route is not None
        assert route.intent == "OPEN_APPLICATION"
        assert route.steps[0].call.tool == "open_application"
        assert route.steps[0].call.arguments == {"app_name": expected_arg}


@patch("os.path.isfile")
@patch("subprocess.Popen")
def test_tool_router_executes_brave_safely(mock_popen, mock_isfile):
    from core.tool_router import ToolCall, ToolRouter
    from core.confirmation import AutoDenyHandler

    mock_isfile.side_effect = lambda p: "brave.exe" in p.lower()
    mock_proc = MagicMock()
    mock_proc.pid = 5555
    mock_popen.return_value = mock_proc

    registry = ToolRegistry()
    register_application_tools(registry)

    # AutoDenyHandler ensures SAFE tools execute without prompting for confirmation
    router = ToolRouter(registry, confirmation_handler=AutoDenyHandler())

    call = ToolCall(tool="open_application", arguments={"app_name": "brave browser"})
    result = router.execute_tool_call(call)

    assert result.success is True
    assert result.message == "Opened Brave Browser."
    assert result.data["app_key"] == "brave"
    mock_popen.assert_called_once()
    assert mock_popen.call_args[1].get("shell") is not True


@patch("subprocess.Popen")
def test_notepad_still_launches_safely(mock_popen):
    mock_proc = MagicMock()
    mock_proc.pid = 7777
    mock_popen.return_value = mock_proc

    tool = OpenApplicationTool()
    result = tool.run(app_name="notepad")

    assert result.success is True
    assert result.message == "Opened Notepad."
    assert result.data["app_key"] == "notepad"
    mock_popen.assert_called_once()

