from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from core.tool_router import ToolCall, ToolRouter
from core.assistant import Assistant, _explicit_application_request
from tools.applications import OpenApplicationTool
from tools.base import Tool, ToolResult
from tools.files import (
    DeleteFileTool,
    ListFilesTool,
    MoveFileTool,
    RenameFileTool,
    SearchFilesTool,
    GetFileInfoTool,
    CreateFolderTool,
)
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry


class Confirmation:
    def __init__(self, approved: bool):
        self.approved = approved
        self.calls = 0

    def request_confirmation(self, **kwargs):
        self.calls += 1
        return self.approved


def _router(tool: Tool, approved: bool = True):
    registry = ToolRegistry()
    registry.register(tool)
    confirmation = Confirmation(approved)
    return ToolRouter(registry, confirmation), confirmation


@pytest.mark.parametrize("tool,args", [
    (ListFilesTool(), {"path": "x", "recursive": "yes"}),
    (ListFilesTool(), {"path": "x", "alien": True}),
    (SearchFilesTool(), {"directory": "x", "query": "a", "max_results": True}),
    (GetFileInfoTool(), {"path": 12}),
])
def test_invalid_file_tool_args_are_rejected_before_execution(tool, args, tmp_path):
    # Replace execute to prove that validation blocks dispatch.
    tool.execute = Mock(return_value=ToolResult.ok("unexpected"))
    router, _ = _router(tool)
    result = router.execute_tool_call(ToolCall(tool.name, args))
    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"
    tool.execute.assert_not_called()


def test_missing_required_arguments_do_not_execute():
    tool = GetFileInfoTool()
    tool.execute = Mock(return_value=ToolResult.ok("unexpected"))
    router, _ = _router(tool)
    result = router.execute_tool_call(ToolCall("get_file_info", {}))
    assert result.error_code == "INVALID_ARGUMENTS"
    tool.execute.assert_not_called()


def test_invalid_dangerous_args_do_not_prompt_or_execute(tmp_path):
    tool = DeleteFileTool()
    tool.execute = Mock(return_value=ToolResult.ok("unexpected"))
    router, confirmation = _router(tool, approved=True)
    result = router.execute_tool_call(ToolCall("delete_file", {"path": str(tmp_path / "x"), "extra": 1}))
    assert result.error_code == "INVALID_ARGUMENTS"
    assert confirmation.calls == 0
    tool.execute.assert_not_called()


def test_confirmation_deny_and_approve_control_disposable_rename(tmp_path):
    target = tmp_path / "before.txt"
    target.write_text("data", encoding="utf-8")
    router, confirmation = _router(RenameFileTool(), approved=False)
    denied = router.execute_tool_call(ToolCall("rename_file", {"path": str(target), "new_name": "after.txt"}))
    assert denied.error_code == "PERMISSION_DENIED"
    assert target.exists()
    assert not (tmp_path / "after.txt").exists()
    assert confirmation.calls == 1

    router, confirmation = _router(RenameFileTool(), approved=True)
    approved = router.execute_tool_call(ToolCall("rename_file", {"path": str(target), "new_name": "after.txt"}))
    assert approved.success
    assert not target.exists()
    assert (tmp_path / "after.txt").exists()
    assert confirmation.calls == 1


def test_create_folder_denial_and_approval_are_enforced(tmp_path):
    denied_path = tmp_path / "denied-folder"
    router, confirmation = _router(CreateFolderTool(), approved=False)
    denied = router.execute_tool_call(ToolCall("create_folder", {"path": str(denied_path)}))
    assert denied.error_code == "PERMISSION_DENIED"
    assert not denied_path.exists()
    assert confirmation.calls == 1

    approved_path = tmp_path / "approved-folder"
    router, confirmation = _router(CreateFolderTool(), approved=True)
    approved = router.execute_tool_call(ToolCall("create_folder", {"path": str(approved_path)}))
    assert approved.success and approved_path.is_dir()
    assert confirmation.calls == 1


def test_move_file_denial_and_approval_are_enforced(tmp_path):
    denied_source = tmp_path / "denied-source.txt"
    denied_target = tmp_path / "denied-target.txt"
    denied_source.write_text("keep", encoding="utf-8")
    router, confirmation = _router(MoveFileTool(), approved=False)
    denied = router.execute_tool_call(ToolCall(
        "move_file", {"source": str(denied_source), "destination": str(denied_target)}
    ))
    assert denied.error_code == "PERMISSION_DENIED"
    assert denied_source.exists() and not denied_target.exists()
    assert confirmation.calls == 1

    approved_source = tmp_path / "approved-source.txt"
    approved_target = tmp_path / "approved-target.txt"
    approved_source.write_text("move", encoding="utf-8")
    router, confirmation = _router(MoveFileTool(), approved=True)
    approved = router.execute_tool_call(ToolCall(
        "move_file", {"source": str(approved_source), "destination": str(approved_target)}
    ))
    assert approved.success
    assert not approved_source.exists() and approved_target.read_text(encoding="utf-8") == "move"
    assert confirmation.calls == 1


def test_delete_file_denial_and_approval_are_enforced(tmp_path):
    denied_target = tmp_path / "keep.txt"
    denied_target.write_text("keep", encoding="utf-8")
    router, confirmation = _router(DeleteFileTool(), approved=False)
    denied = router.execute_tool_call(ToolCall("delete_file", {"path": str(denied_target)}))
    assert denied.error_code == "PERMISSION_DENIED"
    assert denied_target.exists()
    assert confirmation.calls == 1

    approved_target = tmp_path / "delete.txt"
    approved_target.write_text("dispose", encoding="utf-8")
    router, confirmation = _router(DeleteFileTool(), approved=True)
    approved = router.execute_tool_call(ToolCall("delete_file", {"path": str(approved_target)}))
    assert approved.success and not approved_target.exists()
    assert confirmation.calls == 1

def test_list_and_search_results_are_bounded_and_invalid_limits_fail(tmp_path):
    for i in range(12):
        (tmp_path / f"item-{i}.txt").touch()
    listed = ListFilesTool().run(path=str(tmp_path), max_items=500_000)
    searched = SearchFilesTool().run(directory=str(tmp_path), query="item", max_results=500_000)
    assert listed.success and listed.data["total_listed"] == 12
    assert searched.success and searched.data["total_matches"] == 12
    assert ListFilesTool().run(path=str(tmp_path), max_items=0).error_code == "INVALID_ARGUMENTS"
    assert SearchFilesTool().run(directory=str(tmp_path), query="item", max_results=True).error_code == "INVALID_ARGUMENTS"


@pytest.mark.parametrize("tool,args", [
    (ListFilesTool(), {"path": "unused", "recursive": True}),
    (SearchFilesTool(), {"directory": "unused", "query": "item", "recursive": True}),
])
def test_recursive_file_tools_surface_walk_access_errors(tool, args, monkeypatch):
    import tools.files as files_module

    def denied_walk(*walk_args, onerror=None, **walk_kwargs):
        assert onerror is not None
        onerror(PermissionError("access denied"))
        return iter(())

    monkeypatch.setattr(files_module.os, "walk", denied_walk)
    monkeypatch.setattr(files_module, "_resolve_safe_path", lambda _: __import__("pathlib").Path("."))
    result = tool.run(**args)
    assert result.success is False
    assert result.error_code == "PERMISSION_DENIED"


def test_delete_symlink_removes_link_without_deleting_target(tmp_path):
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_text("keep", encoding="utf-8")
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable in this environment")
    result = DeleteFileTool().run(path=str(link))
    assert result.success
    assert not link.exists()
    assert target.read_text(encoding="utf-8") == "keep"


def test_tool_failure_result_has_stable_error_code():
    class Broken(Tool):
        name = "broken"
        description = "Raises during execution"
        permission_level = PermissionLevel.SAFE

        def execute(self, **kwargs):
            raise RuntimeError("opaque failure")

    result = Broken().run()
    assert result.success is False
    assert result.error_code == "EXECUTION_ERROR"
    assert result.to_dict()["error_code"] == "EXECUTION_ERROR"


@pytest.mark.parametrize("prompt,expected", [
    ("Open Notepad", "Notepad"),
    ("Please launch the calculator.", "calculator"),
    ("Could you start VS Code?", "VS Code"),
])
def test_explicit_app_launch_requests_are_narrowly_parsed(prompt, expected):
    assert _explicit_application_request(prompt) == expected


@pytest.mark.parametrize("prompt", [
    "How do I open Notepad?",
    "Open Notepad and run calc.exe; powershell",
    "open C:/Windows/system32/cmd.exe",
])
def test_educational_or_shell_like_app_prompts_are_not_pre_routed(prompt):
    assert _explicit_application_request(prompt) is None


@patch("tools.applications.subprocess.Popen")
def test_text_assistant_launches_allowlisted_app_through_router(mock_popen):
    mock_popen.return_value.pid = 4321
    registry = ToolRegistry()
    registry.register(OpenApplicationTool())
    router = ToolRouter(registry)
    provider = MagicMock()
    provider.generate.return_value = "The application is open."
    assistant = Assistant(provider, tool_router=router)

    response = assistant.send("Open Notepad.")

    assert response == "Opened Notepad."
    mock_popen.assert_called_once_with([mock_popen.call_args.args[0][0]])
    assert mock_popen.call_args.args[0][0].lower().endswith("notepad.exe")
    assert assistant.history[-2]["role"] == "user"
    assert "```tool_result" in assistant.history[-2]["content"]


def test_text_assistant_unknown_app_is_rejected_without_process_launch():
    registry = ToolRegistry()
    registry.register(OpenApplicationTool())
    provider = MagicMock()
    provider.generate.return_value = "That application is not allowlisted."
    assistant = Assistant(provider, tool_router=ToolRouter(registry))

    with patch("tools.applications.subprocess.Popen") as mock_popen:
        response = assistant.send("Open powershell")

    assert "not supported" in response.lower() and "not opened" in response.lower()
    mock_popen.assert_not_called()


def test_current_directory_list_request_returns_live_file_tool_result(tmp_path, monkeypatch):
    (tmp_path / "disposable.txt").write_text("safe", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    registry = ToolRegistry()
    from tools.files import ListFilesTool

    registry.register(ListFilesTool())
    provider = MagicMock()
    provider.generate.return_value = "I found your files."
    assistant = Assistant(provider, tool_router=ToolRouter(registry))

    response = assistant.send("List files in my current directory.")

    assert "disposable.txt" in response
    assert "```tool_result" in assistant.history[-2]["content"]
    assert "disposable.txt" in assistant.history[-2]["content"]
