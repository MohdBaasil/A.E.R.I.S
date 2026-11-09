"""
AERIS Tool System.

Provides the base Tool interface, ToolRegistry, PermissionLevel, ToolResult,
and domain-specific tool modules.
"""

from tools.permissions import PermissionLevel
from tools.base import Tool, ToolResult
from tools.registry import ToolRegistry, ToolRegistryError, ToolNotFoundError
from tools.applications import (
    OpenApplicationTool,
    APPLICATION_ALLOWLIST,
    register_application_tools,
)
from tools.websites import (
    OpenWebsiteTool,
    WEBSITE_ALLOWLIST,
    register_website_tools,
)
from tools.system import (
    SystemInfoTool,
    register_system_tools,
)
from tools.system_audio import (
    SystemAudioTool,
    register_system_audio_tools,
)
from tools.system_media import (
    SystemMediaTool,
    register_system_media_tools,
)
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

__all__ = [
    "PermissionLevel",
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolNotFoundError",
    "OpenApplicationTool",
    "APPLICATION_ALLOWLIST",
    "register_application_tools",
    "OpenWebsiteTool",
    "WEBSITE_ALLOWLIST",
    "register_website_tools",
    "SystemInfoTool",
    "register_system_tools",
    "SystemAudioTool",
    "register_system_audio_tools",
    "SystemMediaTool",
    "register_system_media_tools",
    "ListFilesTool",
    "SearchFilesTool",
    "GetFileInfoTool",
    "CreateFolderTool",
    "MoveFileTool",
    "RenameFileTool",
    "DeleteFileTool",
    "register_file_tools",
]
