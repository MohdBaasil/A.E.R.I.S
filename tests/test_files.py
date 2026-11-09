import os
import pathlib
import pytest

from tools.files import (
    ListFilesTool,
    SearchFilesTool,
    GetFileInfoTool,
    CreateFolderTool,
    MoveFileTool,
    RenameFileTool,
    DeleteFileTool,
    register_file_tools,
)
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry


# ==========================================
# 1. list_files Tests
# ==========================================

def test_list_files_normal_and_empty(tmp_path: pathlib.Path):
    tool = ListFilesTool()
    assert tool.permission_level == PermissionLevel.SAFE

    # Empty directory
    res_empty = tool.run(path=str(tmp_path))
    assert res_empty.success is True
    assert res_empty.data["total_listed"] == 0

    # Create files and subfolder
    (tmp_path / "file1.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "file2.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "subdir").mkdir()

    res = tool.run(path=str(tmp_path))
    assert res.success is True
    assert res.data["total_listed"] == 3
    names = [item["name"] for item in res.data["items"]]
    assert "file1.txt" in names
    assert "file2.py" in names
    assert "subdir" in names


def test_list_files_recursive(tmp_path: pathlib.Path):
    tool = ListFilesTool()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested", encoding="utf-8")

    res = tool.run(path=str(tmp_path), recursive=True)
    assert res.success is True
    names = [item["name"] for item in res.data["items"]]
    assert "sub" in names
    assert "nested.txt" in names


def test_list_files_nonexistent_and_file(tmp_path: pathlib.Path):
    tool = ListFilesTool()
    
    # Nonexistent path
    res_notfound = tool.run(path=str(tmp_path / "does_not_exist"))
    assert res_notfound.success is False
    assert res_notfound.error == "PATH_NOT_FOUND"

    # File path instead of directory
    f = tmp_path / "some_file.txt"
    f.write_text("abc", encoding="utf-8")
    res_file = tool.run(path=str(f))
    assert res_file.success is False
    assert res_file.error == "NOT_A_DIRECTORY"


# ==========================================
# 2. search_files Tests
# ==========================================

def test_search_files_matching_and_no_match(tmp_path: pathlib.Path):
    tool = SearchFilesTool()
    assert tool.permission_level == PermissionLevel.SAFE

    (tmp_path / "project_notes.txt").write_text("notes", encoding="utf-8")
    (tmp_path / "code.py").write_text("code", encoding="utf-8")

    # Match
    res_match = tool.run(directory=str(tmp_path), query="project")
    assert res_match.success is True
    assert res_match.data["total_matches"] == 1
    assert res_match.data["matches"][0]["name"] == "project_notes.txt"

    # No match
    res_nomatch = tool.run(directory=str(tmp_path), query="missing_term")
    assert res_nomatch.success is True
    assert res_nomatch.data["total_matches"] == 0


def test_search_files_recursive(tmp_path: pathlib.Path):
    tool = SearchFilesTool()
    deep_dir = tmp_path / "a" / "b"
    deep_dir.mkdir(parents=True)
    (deep_dir / "target_document.pdf").write_text("pdf", encoding="utf-8")

    res = tool.run(directory=str(tmp_path), query="target", recursive=True)
    assert res.success is True
    assert res.data["total_matches"] == 1
    assert res.data["matches"][0]["name"] == "target_document.pdf"


def test_search_files_invalid_path_and_empty_query(tmp_path: pathlib.Path):
    tool = SearchFilesTool()
    
    res_empty = tool.run(directory=str(tmp_path), query="  ")
    assert res_empty.success is False
    assert res_empty.error == "INVALID_QUERY"

    res_missing_dir = tool.run(directory=str(tmp_path / "missing"), query="abc")
    assert res_missing_dir.success is False
    assert res_missing_dir.error == "DIRECTORY_NOT_FOUND"


# ==========================================
# 3. get_file_info Tests
# ==========================================

def test_get_file_info_file_and_directory(tmp_path: pathlib.Path):
    tool = GetFileInfoTool()
    assert tool.permission_level == PermissionLevel.SAFE

    # Test file
    test_file = tmp_path / "report.csv"
    test_file.write_text("col1,col2\nval1,val2", encoding="utf-8")

    res_file = tool.run(path=str(test_file))
    assert res_file.success is True
    assert res_file.data["name"] == "report.csv"
    assert res_file.data["extension"] == ".csv"
    assert res_file.data["type"] == "file"
    assert res_file.data["size_bytes"] > 0
    assert "created_time" in res_file.data
    assert "modified_time" in res_file.data

    # Test directory
    res_dir = tool.run(path=str(tmp_path))
    assert res_dir.success is True
    assert res_dir.data["type"] == "directory"
    assert res_dir.data["extension"] == ""


def test_get_file_info_nonexistent(tmp_path: pathlib.Path):
    tool = GetFileInfoTool()
    res = tool.run(path=str(tmp_path / "nonexistent.txt"))
    assert res.success is False
    assert res.error == "PATH_NOT_FOUND"


# ==========================================
# 4. create_folder Tests
# ==========================================

def test_create_folder_success_and_existing(tmp_path: pathlib.Path):
    tool = CreateFolderTool()
    assert tool.permission_level == PermissionLevel.CONFIRM

    new_folder = tmp_path / "new_project"
    res_create = tool.run(path=str(new_folder))
    assert res_create.success is True
    assert new_folder.exists()
    assert new_folder.is_dir()

    # Existing folder should fail with ALREADY_EXISTS
    res_duplicate = tool.run(path=str(new_folder))
    assert res_duplicate.success is False
    assert res_duplicate.error == "ALREADY_EXISTS"


# ==========================================
# 5. move_file Tests
# ==========================================

def test_move_file_success(tmp_path: pathlib.Path):
    tool = MoveFileTool()
    assert tool.permission_level == PermissionLevel.CONFIRM

    src = tmp_path / "source.txt"
    src.write_text("move me", encoding="utf-8")
    dst = tmp_path / "dest.txt"

    res = tool.run(source=str(src), destination=str(dst))
    assert res.success is True
    assert not src.exists()
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "move me"


def test_move_file_into_directory(tmp_path: pathlib.Path):
    tool = MoveFileTool()
    src = tmp_path / "doc.txt"
    src.write_text("content", encoding="utf-8")
    dest_dir = tmp_path / "subfolder"
    dest_dir.mkdir()

    res = tool.run(source=str(src), destination=str(dest_dir))
    assert res.success is True
    assert not src.exists()
    assert (dest_dir / "doc.txt").exists()


def test_move_file_missing_source_and_destination_conflict(tmp_path: pathlib.Path):
    tool = MoveFileTool()

    # Missing source
    res_nosrc = tool.run(source=str(tmp_path / "ghost.txt"), destination=str(tmp_path / "out.txt"))
    assert res_nosrc.success is False
    assert res_nosrc.error == "SOURCE_NOT_FOUND"

    # Destination conflict
    src = tmp_path / "a.txt"
    dst = tmp_path / "b.txt"
    src.write_text("a", encoding="utf-8")
    dst.write_text("b", encoding="utf-8")

    res_conflict = tool.run(source=str(src), destination=str(dst))
    assert res_conflict.success is False
    assert res_conflict.error == "DESTINATION_EXISTS"
    assert src.exists()  # Source remains intact


# ==========================================
# 6. rename_file Tests & Path Traversal Security
# ==========================================

def test_rename_file_success(tmp_path: pathlib.Path):
    tool = RenameFileTool()
    assert tool.permission_level == PermissionLevel.CONFIRM

    src = tmp_path / "old_name.txt"
    src.write_text("rename me", encoding="utf-8")

    res = tool.run(path=str(src), new_name="new_name.txt")
    assert res.success is True
    assert not src.exists()
    assert (tmp_path / "new_name.txt").exists()


def test_rename_file_security_rejections(tmp_path: pathlib.Path):
    tool = RenameFileTool()
    src = tmp_path / "safe.txt"
    src.write_text("data", encoding="utf-8")

    # Reject relative traversal
    for bad_name in ["../escape.txt", "..\\escape.txt", "sub/folder.txt", "sub\\folder.txt"]:
        res = tool.run(path=str(src), new_name=bad_name)
        assert res.success is False
        assert res.error == "INVALID_NAME_TRAVERSAL"
        assert src.exists()

    # Reject absolute/drive paths
    res_abs = tool.run(path=str(src), new_name="C:\\Windows\\evil.txt")
    assert res_abs.success is False
    assert res_abs.error == "INVALID_NAME_TRAVERSAL"


def test_rename_file_destination_conflict(tmp_path: pathlib.Path):
    tool = RenameFileTool()
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    f1.write_text("1", encoding="utf-8")
    f2.write_text("2", encoding="utf-8")

    res = tool.run(path=str(f1), new_name="file2.txt")
    assert res.success is False
    assert res.error == "DESTINATION_EXISTS"
    assert f1.exists()


# ==========================================
# 7. delete_file Tests & Security Guards
# ==========================================

def test_delete_file_success(tmp_path: pathlib.Path):
    tool = DeleteFileTool()
    assert tool.permission_level == PermissionLevel.DANGEROUS

    target = tmp_path / "delete_me.tmp"
    target.write_text("temp", encoding="utf-8")

    res = tool.run(path=str(target))
    assert res.success is True
    assert not target.exists()


def test_delete_file_wildcard_rejection(tmp_path: pathlib.Path):
    tool = DeleteFileTool()

    # Strictly reject wildcard patterns
    for wildcard in ["*.txt", "*.*", "temp_?.log", "C:\\*"]:
        res = tool.run(path=wildcard)
        assert res.success is False
        assert res.error == "WILDCARD_NOT_ALLOWED"


def test_delete_file_directory_rejection(tmp_path: pathlib.Path):
    tool = DeleteFileTool()
    folder = tmp_path / "important_folder"
    folder.mkdir()

    # Must refuse to delete directories
    res = tool.run(path=str(folder))
    assert res.success is False
    assert res.error == "DIRECTORY_DELETION_NOT_SUPPORTED"
    assert folder.exists()


def test_delete_file_nonexistent(tmp_path: pathlib.Path):
    tool = DeleteFileTool()
    res = tool.run(path=str(tmp_path / "ghost.txt"))
    assert res.success is False
    assert res.error == "FILE_NOT_FOUND"


# ==========================================
# Registry Integration Tests
# ==========================================

def test_register_file_tools_all_registered():
    registry = ToolRegistry()
    register_file_tools(registry)

    expected_tools = [
        "list_files",
        "search_files",
        "get_file_info",
        "create_folder",
        "move_file",
        "rename_file",
        "delete_file",
    ]

    for name in expected_tools:
        assert registry.has(name) is True
        tool = registry.get(name)
        assert tool is not None
        assert tool.name == name
