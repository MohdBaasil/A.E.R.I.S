"""
System Media Playback Control Tool for AERIS.

Provides safe Windows media playback adjustments via virtual multimedia keys:
- play_pause: Dispatches VK_MEDIA_PLAY_PAUSE key event (toggle semantics).
- next_track: Dispatches VK_MEDIA_NEXT_TRACK key event.
- previous_track: Dispatches VK_MEDIA_PREV_TRACK key event.

Note: These actions dispatch standard Windows keyboard media events to the
currently active media session. They do not query or guarantee playback state.
"""

from __future__ import annotations

import ctypes
import logging
import platform
from typing import Any, Callable, Dict, Optional

from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry

logger = logging.getLogger("aeris.tools.system_media")

# Windows virtual key codes for multimedia controls
VK_MEDIA_NEXT_TRACK = 0xB0  # 176
VK_MEDIA_PREV_TRACK = 0xB1  # 177
VK_MEDIA_PLAY_PAUSE = 0xB3  # 179

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

# Allowed actions for validation
ALLOWED_ACTIONS: frozenset[str] = frozenset({"play_pause", "next_track", "previous_track"})

# Virtual keys map
MEDIA_KEY_MAP: Dict[str, int] = {
    "play_pause": VK_MEDIA_PLAY_PAUSE,
    "next_track": VK_MEDIA_NEXT_TRACK,
    "previous_track": VK_MEDIA_PREV_TRACK,
}

ACTION_MESSAGES: Dict[str, str] = {
    "play_pause": "Dispatched play/pause media key event.",
    "next_track": "Dispatched next track media key event.",
    "previous_track": "Dispatched previous track media key event.",
}


def _send_virtual_key(vk_code: int) -> None:
    """Send keydown and keyup events for a Windows virtual key."""
    if not hasattr(ctypes, "windll"):
        raise OSError("Windows user32 library is unavailable on this platform.")
    user32 = ctypes.windll.user32
    user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
    user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


class SystemMediaTool(Tool):
    """
    Tool to dispatch Windows media playback control key events.

    Supported actions:
        - play_pause: Toggles active media playback (play/pause).
        - next_track: Requests next track in active media queue.
        - previous_track: Requests previous track in active media queue.

    Note: These actions simulate Windows multimedia key events. They do not
    guarantee that an active media player is open or confirm playback state changes.
    """
    name = "system_media"
    description = (
        "Controls Windows media playback using key events: 'play_pause', 'next_track', "
        "or 'previous_track'. Dispatches key events to the active media session."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Media action to perform: 'play_pause', 'next_track', or 'previous_track'.",
                "enum": ["play_pause", "next_track", "previous_track"],
            }
        },
        "required": ["action"],
    }
    permission_level = PermissionLevel.SAFE

    def __init__(self, key_sender: Optional[Callable[[int], None]] = None) -> None:
        super().__init__()
        self._key_sender = key_sender or _send_virtual_key

    def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        """
        Execute media playback control key dispatch.

        Args:
            action: One of 'play_pause', 'next_track', or 'previous_track'.
        """
        if platform.system() != "Windows":
            return ToolResult.fail(
                error="UNSUPPORTED_PLATFORM",
                message="System media control is only supported on Windows.",
                error_code="UNSUPPORTED_PLATFORM",
            )

        if not action or not isinstance(action, str) or action not in ALLOWED_ACTIONS:
            allowed = ", ".join(sorted(ALLOWED_ACTIONS))
            return ToolResult.fail(
                error="INVALID_ACTION",
                message=f"Action '{action}' is not supported. Must be one of: {allowed}.",
                error_code="INVALID_ARGUMENTS",
            )

        vk_code = MEDIA_KEY_MAP[action]
        try:
            self._key_sender(vk_code)
        except Exception as exc:
            logger.error("Failed to dispatch media key event for '%s' (%s)", action, type(exc).__name__)
            return ToolResult.fail(
                error=str(exc),
                message=f"Failed to execute media action '{action}': {exc}",
                error_code="EXECUTION_ERROR",
            )

        message = ACTION_MESSAGES.get(action, f"Dispatched media action '{action}'.")
        return ToolResult.ok(
            message=message,
            data={"action": action, "vk_code": vk_code},
        )


def register_system_media_tools(registry: ToolRegistry) -> None:
    """Register system media control tools with a ToolRegistry."""
    registry.register(SystemMediaTool(), overwrite=True)
