"""
File Management Tools for AERIS.

Provides secure, structured filesystem operations:
- list_files (SAFE)
- search_files (SAFE)
- get_file_info (SAFE)
- create_folder (CONFIRM)
- move_file (CONFIRM)
- rename_file (CONFIRM)
- delete_file (DANGEROUS)

Strictly uses Python standard library (pathlib, os, shutil).
No shell execution or arbitrary wildcard deletion.
"""

from __future__ import annotations

import datetime
import logging
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional

from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry

logger = logging.getLogger("aeris.tools.files")
MAX_FILE_RESULTS = 1000


def _resolve_safe_path(raw_path: str) -> pathlib.Path:
    """
    Safely expand user and environment variables and resolve to an absolute Path.
    """
    if not raw_path or not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Path must be a non-empty string.")
    
    expanded = os.path.expandvars(os.path.expanduser(raw_path.strip()))
    # Normalize dot segments without dereferencing the final symlink. This
    # ensures unlink/rename act on the path the user named, not its target.
    return pathlib.Path(os.path.abspath(expanded))


def _is_link_or_junction(path: pathlib.Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


def _raise_walk_error(error: OSError) -> None:
    """Do not silently report incomplete recursive results on access errors."""
    raise error


def _bounded_limit(value: Any) -> Optional[int]:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return None
    return min(value, MAX_FILE_RESULTS)


# ==============================================================================
# 1. list_files Tool (SAFE)
# ==============================================================================

class ListFilesTool(Tool):
    """Tool to inspect files and directories within a specified path."""
    name = "list_files"
    description = (
        "Lists files and folders in a specified directory path. "
        "Returns names, types (file/directory), and sizes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory to list.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to list subdirectories recursively (default False).",
                "default": False,
            },
            "max_items": {
                "type": "integer",
                "description": "Maximum number of items to return (default 100).",
                "default": 100,
            },
        },
        "required": ["path"],
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, path: str = "", recursive: bool = False, max_items: int = 100, **kwargs: Any) -> ToolResult:
        if not isinstance(recursive, bool):
            return ToolResult.fail("INVALID_ARGUMENTS", message="recursive must be a boolean.")
        max_items = _bounded_limit(max_items)
        if max_items is None:
            return ToolResult.fail("INVALID_ARGUMENTS", message="max_items must be a positive integer.")
        try:
            target = _resolve_safe_path(path)
        except Exception as exc:
            return ToolResult.fail(error="INVALID_PATH", message=f"Invalid path argument: {exc}")

        if not target.exists():
            return ToolResult.fail(
                error="PATH_NOT_FOUND",
                message=f"Directory '{target}' does not exist.",
                data={"path": str(target)},
            )

        if not target.is_dir():
            return ToolResult.fail(
                error="NOT_A_DIRECTORY",
                message=f"Path '{target}' is a file, not a directory.",
                data={"path": str(target)},
            )

        items: List[Dict[str, Any]] = []
        try:
            if recursive:
                for root, dirs, files in os.walk(target, followlinks=False, onerror=_raise_walk_error):
                    dirs[:] = [d for d in dirs if not _is_link_or_junction(pathlib.Path(root) / d)]
                    rel_root = os.path.relpath(root, target)
                    if rel_root != ".":
                        items.append({
                            "name": os.path.basename(root),
                            "relative_path": rel_root,
                            "type": "directory",
                        })
                        if len(items) >= max_items:
                            break
                    for f in files:
                        file_path = os.path.join(root, f)
                        try:
                            size = os.path.getsize(file_path)
                        except OSError:
                            size = None
                        items.append({
                            "name": f,
                            "relative_path": os.path.relpath(file_path, target),
                            "type": "file",
                            "size_bytes": size,
                        })
                        if len(items) >= max_items:
                            break
                    if len(items) >= max_items:
                        break
            else:
                with os.scandir(target) as scanner:
                    for entry in scanner:
                        try:
                            is_file = entry.is_file(follow_symlinks=False)
                            is_dir = entry.is_dir(follow_symlinks=False)
                            entry_type = "file" if is_file else ("directory" if is_dir else "other")
                            size = entry.stat(follow_symlinks=False).st_size if is_file else None
                        except OSError:
                            entry_type = "unknown"
                            size = None

                        items.append({
                            "name": entry.name,
                            "relative_path": entry.name,
                            "type": entry_type,
                            "size_bytes": size,
                        })
                        if len(items) >= max_items:
                            break

            return ToolResult.ok(
                message=f"Listed {len(items)} item(s) in '{target}'.",
                data={
                    "path": str(target),
                    "total_listed": len(items),
                    "items": items,
                },
            )

        except PermissionError as exc:
            logger.error("Permission denied listing directory '%s': %s", target, exc)
            return ToolResult.fail(
                error="PERMISSION_DENIED",
                message=f"Access denied reading directory '{target}'.",
                data={"path": str(target)},
            )
        except Exception as exc:
            logger.exception("Unexpected error listing directory '%s': %s", target, exc)
            return ToolResult.fail(
                error="LIST_FAILED",
                message=f"Failed to list directory '{target}': {exc}",
                data={"path": str(target)},
            )


# ==============================================================================
# 2. search_files Tool (SAFE)
# ==============================================================================

class SearchFilesTool(Tool):
    """Tool to search for files matching a name/pattern in a directory."""
    name = "search_files"
    description = (
        "Searches for files and directories matching a query name inside a directory. "
        "Non-recursive by default."
    )
    parameters = {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Base directory to search in.",
            },
            "query": {
                "type": "string",
                "description": "Text or filename substring to match (case-insensitive).",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to search subdirectories recursively (default False).",
                "default": False,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return (default 50).",
                "default": 50,
            },
        },
        "required": ["directory", "query"],
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, directory: str = "", query: str = "", recursive: bool = False, max_results: int = 50, **kwargs: Any) -> ToolResult:
        if not isinstance(recursive, bool):
            return ToolResult.fail("INVALID_ARGUMENTS", message="recursive must be a boolean.")
        max_results = _bounded_limit(max_results)
        if max_results is None:
            return ToolResult.fail("INVALID_ARGUMENTS", message="max_results must be a positive integer.")
        if not query or not isinstance(query, str) or not query.strip():
            return ToolResult.fail(
                error="INVALID_QUERY",
                message="Search query must be a non-empty string.",
            )

        try:
            base_dir = _resolve_safe_path(directory)
        except Exception as exc:
            return ToolResult.fail(error="INVALID_PATH", message=f"Invalid directory path: {exc}")

        if not base_dir.exists():
            return ToolResult.fail(
                error="DIRECTORY_NOT_FOUND",
                message=f"Directory '{base_dir}' does not exist.",
                data={"directory": str(base_dir)},
            )

        if not base_dir.is_dir():
            return ToolResult.fail(
                error="NOT_A_DIRECTORY",
                message=f"Path '{base_dir}' is a file, not a directory.",
                data={"directory": str(base_dir)},
            )

        query_normalized = query.strip().lower()
        matches: List[Dict[str, Any]] = []

        try:
            if recursive:
                for root, dirs, files in os.walk(base_dir, followlinks=False, onerror=_raise_walk_error):
                    dirs[:] = [d for d in dirs if not _is_link_or_junction(pathlib.Path(root) / d)]
                    for d in dirs:
                        if query_normalized in d.lower():
                            full_p = os.path.join(root, d)
                            matches.append({
                                "name": d,
                                "path": full_p,
                                "relative_path": os.path.relpath(full_p, base_dir),
                                "type": "directory",
                            })
                            if len(matches) >= max_results:
                                break
                    if len(matches) >= max_results:
                        break

                    for f in files:
                        if query_normalized in f.lower():
                            full_p = os.path.join(root, f)
                            matches.append({
                                "name": f,
                                "path": full_p,
                                "relative_path": os.path.relpath(full_p, base_dir),
                                "type": "file",
                            })
                            if len(matches) >= max_results:
                                break
                    if len(matches) >= max_results:
                        break
            else:
                with os.scandir(base_dir) as scanner:
                    for entry in scanner:
                        if query_normalized in entry.name.lower():
                            try:
                                is_file = entry.is_file(follow_symlinks=False)
                                is_dir = entry.is_dir(follow_symlinks=False)
                                entry_type = "file" if is_file else ("directory" if is_dir else "other")
                            except OSError:
                                entry_type = "unknown"

                            matches.append({
                                "name": entry.name,
                                "path": entry.path,
                                "relative_path": entry.name,
                                "type": entry_type,
                            })
                            if len(matches) >= max_results:
                                break

            return ToolResult.ok(
                message=f"Found {len(matches)} match(es) for '{query}' in '{base_dir}'.",
                data={
                    "directory": str(base_dir),
                    "query": query,
                    "total_matches": len(matches),
                    "matches": matches,
                },
            )

        except PermissionError as exc:
            logger.error("Permission denied searching directory '%s': %s", base_dir, exc)
            return ToolResult.fail(
                error="PERMISSION_DENIED",
                message=f"Access denied while searching '{base_dir}'.",
                data={"directory": str(base_dir)},
            )
        except Exception as exc:
            logger.exception("Unexpected error searching directory '%s': %s", base_dir, exc)
            return ToolResult.fail(
                error="SEARCH_FAILED",
                message=f"Search failed in '{base_dir}': {exc}",
                data={"directory": str(base_dir)},
            )


# ==============================================================================
# 3. get_file_info Tool (SAFE)
# ==============================================================================

class GetFileInfoTool(Tool):
    """Tool to inspect detailed metadata of a file or folder."""
    name = "get_file_info"
    description = (
        "Retrieves detailed metadata for a file or folder (size, type, timestamps, absolute path)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory.",
            }
        },
        "required": ["path"],
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, path: str = "", **kwargs: Any) -> ToolResult:
        try:
            target = _resolve_safe_path(path)
        except Exception as exc:
            return ToolResult.fail(error="INVALID_PATH", message=f"Invalid path argument: {exc}")

        if not target.exists():
            return ToolResult.fail(
                error="PATH_NOT_FOUND",
                message=f"Path '{target}' does not exist.",
                data={"path": str(target)},
            )

        try:
            stat_info = target.stat(follow_symlinks=False)
            is_file = target.is_file()
            is_dir = target.is_dir()
            size_bytes = stat_info.st_size if is_file else None
            size_mb = round(size_bytes / (1024 * 1024), 3) if size_bytes is not None else None

            created_time = datetime.datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            modified_time = datetime.datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            accessed_time = datetime.datetime.fromtimestamp(stat_info.st_atime).isoformat()

            metadata = {
                "absolute_path": str(target),
                "name": target.name,
                "extension": target.suffix if is_file else "",
                "type": "file" if is_file else ("directory" if is_dir else "other"),
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "created_time": created_time,
                "modified_time": modified_time,
                "accessed_time": accessed_time,
                "is_readonly": not os.access(target, os.W_OK),
            }

            return ToolResult.ok(
                message=f"Retrieved metadata for '{target.name}'.",
                data=metadata,
            )

        except PermissionError as exc:
            logger.error("Permission denied inspecting path '%s': %s", target, exc)
            return ToolResult.fail(
                error="PERMISSION_DENIED",
                message=f"Access denied reading metadata for '{target}'.",
                data={"path": str(target)},
            )
        except Exception as exc:
            logger.exception("Unexpected error inspecting path '%s': %s", target, exc)
            return ToolResult.fail(
                error="INSPECT_FAILED",
                message=f"Failed to inspect path '{target}': {exc}",
                data={"path": str(target)},
            )


# ==============================================================================
# 4. create_folder Tool (CONFIRM)
# ==============================================================================

class CreateFolderTool(Tool):
    """Tool to create a new folder."""
    name = "create_folder"
    description = (
        "Creates a new directory. Does not overwrite existing files or folders."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path of the folder to create.",
            }
        },
        "required": ["path"],
    }
    permission_level = PermissionLevel.CONFIRM

    def execute(self, path: str = "", **kwargs: Any) -> ToolResult:
        try:
            target = _resolve_safe_path(path)
        except Exception as exc:
            return ToolResult.fail(error="INVALID_PATH", message=f"Invalid path argument: {exc}")

        if target.exists():
            return ToolResult.fail(
                error="ALREADY_EXISTS",
                message=f"A file or folder already exists at '{target}'.",
                data={"path": str(target)},
            )

        try:
            target.mkdir(parents=True, exist_ok=False)
            return ToolResult.ok(
                message=f"Folder created at '{target}'.",
                data={"path": str(target)},
            )
        except PermissionError as exc:
            logger.error("Permission denied creating folder '%s': %s", target, exc)
            return ToolResult.fail(
                error="PERMISSION_DENIED",
                message=f"Permission denied creating folder at '{target}'.",
                data={"path": str(target)},
            )
        except Exception as exc:
            logger.exception("Unexpected error creating folder '%s': %s", target, exc)
            return ToolResult.fail(
                error="CREATE_FAILED",
                message=f"Failed to create folder '{target}': {exc}",
                data={"path": str(target)},
            )


# ==============================================================================
# 5. move_file Tool (CONFIRM)
# ==============================================================================

class MoveFileTool(Tool):
    """Tool to move a file or directory from source to destination."""
    name = "move_file"
    description = (
        "Moves a file or directory from source path to destination path. "
        "Does not overwrite existing files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Source path of the file or folder to move.",
            },
            "destination": {
                "type": "string",
                "description": "Destination path or target directory.",
            },
        },
        "required": ["source", "destination"],
    }
    permission_level = PermissionLevel.CONFIRM

    def execute(self, source: str = "", destination: str = "", **kwargs: Any) -> ToolResult:
        try:
            src = _resolve_safe_path(source)
            dst = _resolve_safe_path(destination)
        except Exception as exc:
            return ToolResult.fail(error="INVALID_PATH", message=f"Invalid path argument: {exc}")

        if not src.exists():
            return ToolResult.fail(
                error="SOURCE_NOT_FOUND",
                message=f"Source '{src}' does not exist.",
                data={"source": str(src)},
            )

        # If destination is an existing directory, target is inside it
        final_dst = dst / src.name if dst.is_dir() else dst

        if final_dst.exists():
            return ToolResult.fail(
                error="DESTINATION_EXISTS",
                message=f"Destination '{final_dst}' already exists. Move aborted to prevent overwrite.",
                data={"source": str(src), "destination": str(final_dst)},
            )

        try:
            shutil.move(str(src), str(final_dst))
            return ToolResult.ok(
                message=f"Moved '{src.name}' to '{final_dst}'.",
                data={"source": str(src), "destination": str(final_dst)},
            )
        except PermissionError as exc:
            logger.error("Permission denied moving '%s' to '%s': %s", src, final_dst, exc)
            return ToolResult.fail(
                error="PERMISSION_DENIED",
                message=f"Permission denied moving '{src}' to '{final_dst}'.",
                data={"source": str(src), "destination": str(final_dst)},
            )
        except Exception as exc:
            logger.exception("Unexpected error moving '%s' to '%s': %s", src, final_dst, exc)
            return ToolResult.fail(
                error="MOVE_FAILED",
                message=f"Failed to move '{src}' to '{final_dst}': {exc}",
                data={"source": str(src), "destination": str(final_dst)},
            )


# ==============================================================================
# 6. rename_file Tool (CONFIRM)
# ==============================================================================

class RenameFileTool(Tool):
    """Tool to rename a file or folder inside its current directory."""
    name = "rename_file"
    description = (
        "Renames a file or directory within its current parent folder. "
        "The new name must be a single filename (not a path)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory to rename.",
            },
            "new_name": {
                "type": "string",
                "description": "New filename (must NOT contain slashes, path separators, or relative pathing).",
            },
        },
        "required": ["path", "new_name"],
    }
    permission_level = PermissionLevel.CONFIRM

    def execute(self, path: str = "", new_name: str = "", **kwargs: Any) -> ToolResult:
        if not new_name or not isinstance(new_name, str) or not new_name.strip():
            return ToolResult.fail(
                error="INVALID_NAME",
                message="New name must be a non-empty string.",
            )

        clean_name = new_name.strip()

        # Strict security validation: new_name must be a simple filename, no path traversal or separators
        if any(sep in clean_name for sep in ["/", "\\", ":", "..", os.sep, os.altsep or ""]):
            return ToolResult.fail(
                error="INVALID_NAME_TRAVERSAL",
                message=f"Invalid new name '{clean_name}'. Path separators and relative navigation are not allowed in rename.",
                data={"new_name": clean_name},
            )

        try:
            target = _resolve_safe_path(path)
        except Exception as exc:
            return ToolResult.fail(error="INVALID_PATH", message=f"Invalid path argument: {exc}")

        if not target.exists():
            return ToolResult.fail(
                error="PATH_NOT_FOUND",
                message=f"Target '{target}' does not exist.",
                data={"path": str(target)},
            )

        new_target = target.parent / clean_name
        if new_target.exists():
            return ToolResult.fail(
                error="DESTINATION_EXISTS",
                message=f"A file or folder named '{clean_name}' already exists in '{target.parent}'.",
                data={"path": str(target), "new_path": str(new_target)},
            )

        try:
            target.rename(new_target)
            return ToolResult.ok(
                message=f"Renamed '{target.name}' to '{clean_name}'.",
                data={"old_path": str(target), "new_path": str(new_target)},
            )
        except PermissionError as exc:
            logger.error("Permission denied renaming '%s' to '%s': %s", target, new_target, exc)
            return ToolResult.fail(
                error="PERMISSION_DENIED",
                message=f"Permission denied renaming '{target}' to '{clean_name}'.",
                data={"path": str(target), "new_path": str(new_target)},
            )
        except Exception as exc:
            logger.exception("Unexpected error renaming '%s' to '%s': %s", target, new_target, exc)
            return ToolResult.fail(
                error="RENAME_FAILED",
                message=f"Failed to rename '{target}' to '{clean_name}': {exc}",
                data={"path": str(target), "new_path": str(new_target)},
            )


# ==============================================================================
# 7. delete_file Tool (DANGEROUS)
# ==============================================================================

class DeleteFileTool(Tool):
    """Tool to delete a specific file (requires DANGEROUS permission)."""
    name = "delete_file"
    description = (
        "Deletes a specific single file. Does NOT support directory deletion or wildcards."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the single file to delete.",
            }
        },
        "required": ["path"],
    }
    permission_level = PermissionLevel.DANGEROUS

    def execute(self, path: str = "", **kwargs: Any) -> ToolResult:
        if not path or not isinstance(path, str) or not path.strip():
            return ToolResult.fail(
                error="INVALID_PATH",
                message="Path must be a non-empty string.",
            )

        # Security check: disallow wildcards
        if any(wildcard in path for wildcard in ["*", "?", "<", ">", '"', "|"]):
            return ToolResult.fail(
                error="WILDCARD_NOT_ALLOWED",
                message=f"Wildcard pattern in '{path}' is strictly forbidden for deletion.",
                data={"path": path},
            )

        try:
            target = _resolve_safe_path(path)
        except Exception as exc:
            return ToolResult.fail(error="INVALID_PATH", message=f"Invalid path argument: {exc}")

        if not target.exists():
            return ToolResult.fail(
                error="FILE_NOT_FOUND",
                message=f"File '{target}' does not exist.",
                data={"path": str(target)},
            )

        # Safety restriction: only delete files, never directories
        if target.is_dir():
            return ToolResult.fail(
                error="DIRECTORY_DELETION_NOT_SUPPORTED",
                message=f"Path '{target}' is a directory. Directory deletion is forbidden by delete_file.",
                data={"path": str(target)},
            )

        try:
            target.unlink(missing_ok=False)
            return ToolResult.ok(
                message=f"Deleted file '{target.name}'.",
                data={"deleted_path": str(target)},
            )
        except PermissionError as exc:
            logger.error("Permission denied deleting '%s': %s", target, exc)
            return ToolResult.fail(
                error="PERMISSION_DENIED",
                message=f"Permission denied deleting file '{target}'.",
                data={"path": str(target)},
            )
        except Exception as exc:
            logger.exception("Unexpected error deleting file '%s': %s", target, exc)
            return ToolResult.fail(
                error="DELETE_FAILED",
                message=f"Failed to delete file '{target}': {exc}",
                data={"path": str(target)},
            )


# ==============================================================================
# Registration Function
# ==============================================================================

def register_file_tools(registry: ToolRegistry) -> None:
    """
    Register all file management tools with the provided ToolRegistry.
    """
    registry.register(ListFilesTool(), overwrite=True)
    registry.register(SearchFilesTool(), overwrite=True)
    registry.register(GetFileInfoTool(), overwrite=True)
    registry.register(CreateFolderTool(), overwrite=True)
    registry.register(MoveFileTool(), overwrite=True)
    registry.register(RenameFileTool(), overwrite=True)
    registry.register(DeleteFileTool(), overwrite=True)
