"""
Application Control Tools for AERIS.

Provides secure, allowlist-based launching of local Windows applications.
Does not permit arbitrary command execution.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Set

from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry

logger = logging.getLogger("aeris.tools.applications")


# Explicit allowlist of supported applications
APPLICATION_ALLOWLIST: Dict[str, Dict[str, Any]] = {
    "notepad": {
        "display_name": "Notepad",
        "aliases": {"notepad", "note", "text editor"},
        "executable": "notepad.exe",
        "lookup_paths": [],
    },
    "calculator": {
        "display_name": "Calculator",
        "aliases": {"calculator", "calc"},
        "executable": "calc.exe",
        "lookup_paths": [],
    },
    "explorer": {
        "display_name": "File Explorer",
        "aliases": {"explorer", "file explorer", "files", "my computer"},
        "executable": "explorer.exe",
        "lookup_paths": [],
    },
    "chrome": {
        "display_name": "Google Chrome",
        "aliases": {"chrome", "google chrome", "google-chrome", "browser"},
        "executable": "chrome.exe",
        "lookup_paths": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ],
    },
    "brave": {
        "display_name": "Brave Browser",
        "aliases": {"brave", "brave browser", "brave-browser"},
        "executable": "brave.exe",
        "lookup_paths": [
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        ],
    },
    "vscode": {
        "display_name": "Visual Studio Code",
        "aliases": {"vscode", "vs code", "code", "visual studio code"},
        "executable": "code",
        "lookup_paths": [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
            r"C:\Program Files\Microsoft VS Code\Code.exe",
            r"C:\Program Files (x86)\Microsoft VS Code\Code.exe",
        ],
    },
    "edge": {
        "display_name": "Microsoft Edge",
        "aliases": {"edge", "microsoft edge", "msedge"},
        "executable": "msedge.exe",
        "lookup_paths": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
    },
    "paint": {
        "display_name": "Paint",
        "aliases": {"paint", "mspaint"},
        "executable": "mspaint.exe",
        "lookup_paths": [],
    },
    "task_manager": {
        "display_name": "Task Manager",
        "aliases": {"taskmgr", "task manager"},
        "executable": "taskmgr.exe",
        "lookup_paths": [],
    },
}


class OpenApplicationTool(Tool):
    """
    Tool to safely launch approved Windows applications from an allowlist.
    """
    name = "open_application"
    description = (
        "Safely opens an approved Windows application from the allowlist "
        "(e.g., 'brave', 'chrome', 'vscode', 'notepad', 'calculator', 'explorer')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "Name or alias of the application to open (e.g. 'brave', 'chrome', 'vscode', 'notepad').",
            }
        },
        "required": ["app_name"],
    }
    permission_level = PermissionLevel.SAFE

    def __init__(
        self,
        allowlist: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        self._allowlist = allowlist or APPLICATION_ALLOWLIST
        self._alias_map: Dict[str, str] = {}
        self._build_alias_index()

    def _build_alias_index(self) -> None:
        """Construct a normalized lookup table from alias strings to app keys."""
        self._alias_map.clear()
        for app_key, config in self._allowlist.items():
            self._alias_map[app_key.lower()] = app_key
            for alias in config.get("aliases", []):
                self._alias_map[alias.lower().strip()] = app_key

    def resolve_app(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a user application query to its allowlist configuration.
        Returns None if the application is not allowlisted.
        """
        if not user_query:
            return None

        normalized = user_query.strip().lower()
        app_key = self._alias_map.get(normalized)
        if not app_key:
            return None

        return {
            "app_key": app_key,
            **self._allowlist[app_key],
        }

    def get_supported_aliases(self) -> List[str]:
        """Return a sorted list of all supported application names and aliases."""
        return sorted(list(self._alias_map.keys()))

    @staticmethod
    def _find_windows_app_path(executable: str) -> Optional[str]:
        """
        Query Windows registered App Paths for an allowlisted executable name.
        Guarded to only return verified, existing filesystem executable targets.
        """
        if not executable or os.path.basename(executable) != executable:
            return None
        try:
            import winreg
        except ImportError:
            return None

        sub_key = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{executable}"
        for root_hkey in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(root_hkey, sub_key) as key:
                    raw_path, _ = winreg.QueryValueEx(key, "")
                    if isinstance(raw_path, str) and raw_path.strip():
                        cleaned = raw_path.strip().strip('"').strip("'")
                        if os.path.isfile(cleaned) and os.path.basename(cleaned).lower() == executable.lower():
                            return os.path.abspath(cleaned)
            except OSError:
                continue
        return None

    def _find_executable(self, config: Dict[str, Any]) -> Optional[str]:
        """
        Determine the executable path to launch.
        Checks explicit allowlisted locations. System applications are resolved
        only from Windows System32; PATH and the current directory are not trusted.
        """
        # 1. Check explicit lookup paths
        for path in config.get("lookup_paths", []):
            if path and os.path.isfile(path):
                return os.path.abspath(path)

        executable = config.get("executable", "")
        # The catalog uses bare executable names for built-in Windows tools.
        # Refuse paths, switches, or arbitrary command strings in catalog data.
        if not isinstance(executable, str) or not executable or os.path.basename(executable) != executable:
            return None

        # 2. Check Windows registered App Paths
        app_path_candidate = self._find_windows_app_path(executable)
        if app_path_candidate and os.path.isfile(app_path_candidate):
            return app_path_candidate

        # 3. Check System32 for built-in tools
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        system_candidate = os.path.join(system_root, "System32", executable)
        if os.path.isfile(system_candidate):
            return system_candidate
        return None

    def execute(self, app_name: str = "", **kwargs: Any) -> ToolResult:
        """
        Execute application launch for an allowlisted app.
        
        Args:
            app_name: The application identifier or alias to launch.
        """
        if not app_name or not isinstance(app_name, str) or not app_name.strip():
            return ToolResult.fail(
                error="INVALID_ARGUMENT",
                message="Application name must be a non-empty string.",
            )

        app_info = self.resolve_app(app_name)
        if not app_info:
            logger.warning("Rejected attempt to open a non-allowlisted application name")
            return ToolResult.fail(
                error="UNKNOWN_APPLICATION",
                message=f"Application '{app_name}' is not in the approved allowlist.",
                data={"requested": app_name, "supported": sorted(list(self._allowlist.keys()))},
            )

        executable_target = self._find_executable(app_info)
        display_name = app_info.get("display_name", app_info["app_key"])
        if not executable_target:
            return ToolResult.fail(
                error="NOT_FOUND",
                message=f"Application '{display_name}' executable was not found in an approved location.",
            )

        try:
            logger.info("Launching allowlisted application '%s' using target: '%s'", display_name, executable_target)
            
            # Secure execution: direct execution without shell=True to avoid command injection
            proc = subprocess.Popen([executable_target])

            return ToolResult.ok(
                message=f"Opened {display_name}.",
                data={
                    "app_key": app_info["app_key"],
                    "display_name": display_name,
                    "pid": proc.pid,
                    "target": executable_target,
                },
            )

        except FileNotFoundError as exc:
            logger.error("Executable '%s' for '%s' not found on system: %s", executable_target, display_name, exc)
            return ToolResult.fail(
                error="NOT_FOUND",
                message=f"Application '{display_name}' executable was not found on this system.",
                data={"target": executable_target},
            )
        except OSError as exc:
            logger.error("Failed to launch application '%s': %s", display_name, exc)
            return ToolResult.fail(
                error="LAUNCH_FAILED",
                message=f"Failed to launch '{display_name}': {exc}",
                data={"target": executable_target},
            )
        except Exception as exc:
            logger.exception("Unexpected error launching application '%s': %s", display_name, exc)
            return ToolResult.fail(
                error="UNEXPECTED_ERROR",
                message=f"An unexpected error occurred while launching '{display_name}': {exc}",
            )


def register_application_tools(registry: ToolRegistry) -> None:
    """
    Convenience function to register all application control tools with a registry.
    """
    registry.register(OpenApplicationTool(), overwrite=True)
