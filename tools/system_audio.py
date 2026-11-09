"""
System Audio Control Tool for AERIS.

Provides safe Windows master audio adjustments:
- volume_up and volume_down dispatch virtual media key events.
- mute and unmute query and set master mute state idempotently via Windows Core Audio (IAudioEndpointVolume).
Does not support querying or setting exact volume percentages.
"""

from __future__ import annotations

import ctypes
from ctypes import HRESULT, POINTER, Structure, c_int, c_void_p, wintypes
import logging
import platform
from typing import Any, Callable, Dict, Optional

from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry

logger = logging.getLogger("aeris.tools.system_audio")

# Virtual key codes for Windows multimedia volume adjustment
VK_VOLUME_DOWN = 0xAE   # 174
VK_VOLUME_UP = 0xAF     # 175

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

# Allowed actions for validation
ALLOWED_ACTIONS: frozenset[str] = frozenset({"volume_up", "volume_down", "mute", "unmute"})

# Virtual keys used only for incremental volume stepping (mute/unmute use Core Audio)
VOLUME_KEY_MAP: Dict[str, int] = {
    "volume_up": VK_VOLUME_UP,
    "volume_down": VK_VOLUME_DOWN,
}

# Windows Core Audio COM GUID definitions
class _GUID(Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", wintypes.BYTE * 8),
    ]

    def __init__(self, l: int, w1: int, w2: int, b1: int, b2: int, b3: int, b4: int, b5: int, b6: int, b7: int, b8: int):
        super().__init__(l, w1, w2, (wintypes.BYTE * 8)(b1, b2, b3, b4, b5, b6, b7, b8))

_CLSID_MMDeviceEnumerator = _GUID(0xBCDE0395, 0xE52F, 0x467C, 0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E)
_IID_IMMDeviceEnumerator = _GUID(0xA95664D2, 0x9614, 0x4F35, 0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6)
_IID_IAudioEndpointVolume = _GUID(0x5CDF2C82, 0x841E, 0x4546, 0x97, 0x22, 0x0C, 0xF7, 0x40, 0x78, 0x22, 0x9A)

_CLSCTX_ALL = 23
_E_RENDER = 0
_E_CONSOLE = 0


def _send_virtual_key(vk_code: int) -> None:
    """Send keydown and keyup events for a Windows virtual key."""
    if not hasattr(ctypes, "windll"):
        raise OSError("Windows user32 library is unavailable on this platform.")
    user32 = ctypes.windll.user32
    user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
    user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


def _release_com_ptr(ptr: c_void_p) -> None:
    """Safely release an IUnknown COM interface pointer."""
    if ptr and ptr.value:
        try:
            vtbl = ctypes.cast(ptr, POINTER(POINTER(c_void_p))).contents
            release_proto = ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)
            release_fn = release_proto(vtbl[2])
            release_fn(ptr)
        except Exception:
            pass


def _get_master_mute() -> bool:
    """Query current Windows master audio mute status via Windows Core Audio (MMDevice API)."""
    if not hasattr(ctypes, "windll"):
        raise OSError("Windows ole32 is unavailable on this platform.")

    ole32 = ctypes.windll.ole32
    ole32.CoInitialize(None)
    p_enumerator = c_void_p()
    p_device = c_void_p()
    p_audio_endpoint = c_void_p()

    try:
        hr = ole32.CoCreateInstance(
            ctypes.byref(_CLSID_MMDeviceEnumerator),
            None,
            _CLSCTX_ALL,
            ctypes.byref(_IID_IMMDeviceEnumerator),
            ctypes.byref(p_enumerator),
        )
        if hr != 0:
            raise OSError(f"CoCreateInstance MMDeviceEnumerator failed with HRESULT {hr:#x}")

        vtbl_enum = ctypes.cast(p_enumerator, POINTER(POINTER(c_void_p))).contents
        get_default_endpoint = ctypes.WINFUNCTYPE(HRESULT, c_void_p, c_int, c_int, POINTER(c_void_p))(vtbl_enum[4])
        hr = get_default_endpoint(p_enumerator, _E_RENDER, _E_CONSOLE, ctypes.byref(p_device))
        if hr != 0:
            raise OSError(f"GetDefaultAudioEndpoint failed with HRESULT {hr:#x}")

        vtbl_dev = ctypes.cast(p_device, POINTER(POINTER(c_void_p))).contents
        activate = ctypes.WINFUNCTYPE(
            HRESULT, c_void_p, POINTER(_GUID), wintypes.DWORD, c_void_p, POINTER(c_void_p)
        )(vtbl_dev[3])
        hr = activate(p_device, ctypes.byref(_IID_IAudioEndpointVolume), _CLSCTX_ALL, None, ctypes.byref(p_audio_endpoint))
        if hr != 0:
            raise OSError(f"Activate IAudioEndpointVolume failed with HRESULT {hr:#x}")

        vtbl_vol = ctypes.cast(p_audio_endpoint, POINTER(POINTER(c_void_p))).contents
        get_mute = ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(wintypes.BOOL))(vtbl_vol[15])
        is_muted = wintypes.BOOL()
        hr = get_mute(p_audio_endpoint, ctypes.byref(is_muted))
        if hr != 0:
            raise OSError(f"IAudioEndpointVolume::GetMute failed with HRESULT {hr:#x}")

        return bool(is_muted.value)
    finally:
        _release_com_ptr(p_audio_endpoint)
        _release_com_ptr(p_device)
        _release_com_ptr(p_enumerator)
        ole32.CoUninitialize()


def _set_master_mute(mute: bool) -> None:
    """Set Windows master audio mute status via Windows Core Audio (MMDevice API)."""
    if not hasattr(ctypes, "windll"):
        raise OSError("Windows ole32 is unavailable on this platform.")

    ole32 = ctypes.windll.ole32
    ole32.CoInitialize(None)
    p_enumerator = c_void_p()
    p_device = c_void_p()
    p_audio_endpoint = c_void_p()

    try:
        hr = ole32.CoCreateInstance(
            ctypes.byref(_CLSID_MMDeviceEnumerator),
            None,
            _CLSCTX_ALL,
            ctypes.byref(_IID_IMMDeviceEnumerator),
            ctypes.byref(p_enumerator),
        )
        if hr != 0:
            raise OSError(f"CoCreateInstance MMDeviceEnumerator failed with HRESULT {hr:#x}")

        vtbl_enum = ctypes.cast(p_enumerator, POINTER(POINTER(c_void_p))).contents
        get_default_endpoint = ctypes.WINFUNCTYPE(HRESULT, c_void_p, c_int, c_int, POINTER(c_void_p))(vtbl_enum[4])
        hr = get_default_endpoint(p_enumerator, _E_RENDER, _E_CONSOLE, ctypes.byref(p_device))
        if hr != 0:
            raise OSError(f"GetDefaultAudioEndpoint failed with HRESULT {hr:#x}")

        vtbl_dev = ctypes.cast(p_device, POINTER(POINTER(c_void_p))).contents
        activate = ctypes.WINFUNCTYPE(
            HRESULT, c_void_p, POINTER(_GUID), wintypes.DWORD, c_void_p, POINTER(c_void_p)
        )(vtbl_dev[3])
        hr = activate(p_device, ctypes.byref(_IID_IAudioEndpointVolume), _CLSCTX_ALL, None, ctypes.byref(p_audio_endpoint))
        if hr != 0:
            raise OSError(f"Activate IAudioEndpointVolume failed with HRESULT {hr:#x}")

        vtbl_vol = ctypes.cast(p_audio_endpoint, POINTER(POINTER(c_void_p))).contents
        set_mute = ctypes.WINFUNCTYPE(HRESULT, c_void_p, wintypes.BOOL, c_void_p)(vtbl_vol[14])
        # In COM, hr can be S_OK (0) or S_FALSE (1, if value was already in target state)
        hr = set_mute(p_audio_endpoint, wintypes.BOOL(mute), None)
        if hr not in (0, 1):
            raise OSError(f"IAudioEndpointVolume::SetMute failed with HRESULT {hr:#x}")
    finally:
        _release_com_ptr(p_audio_endpoint)
        _release_com_ptr(p_device)
        _release_com_ptr(p_enumerator)
        ole32.CoUninitialize()


class SystemAudioTool(Tool):
    """
    Tool to control Windows system audio.

    Supported actions:
        - volume_up: Increments system volume via media key event.
        - volume_down: Decrements system volume via media key event.
        - mute: Idempotently mutes system audio (leaves system muted).
        - unmute: Idempotently unmutes system audio (leaves system unmuted).

    Note: volume_up and volume_down simulate Windows multimedia key events.
    mute and unmute reliably query and set master mute state via Windows Core Audio.
    Querying or setting an exact percentage volume is not supported.
    """
    name = "system_audio"
    description = (
        "Controls Windows master audio: 'volume_up', 'volume_down', 'mute', or 'unmute'. "
        "Mute and unmute are idempotent. Does not support exact volume percentages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Audio action to perform: 'volume_up', 'volume_down', 'mute', or 'unmute'.",
                "enum": ["volume_up", "volume_down", "mute", "unmute"],
            }
        },
        "required": ["action"],
    }
    permission_level = PermissionLevel.SAFE

    def __init__(
        self,
        key_sender: Optional[Callable[[int], None]] = None,
        mute_getter: Optional[Callable[[], bool]] = None,
        mute_setter: Optional[Callable[[bool], None]] = None,
    ) -> None:
        super().__init__()
        self._key_sender = key_sender or _send_virtual_key
        self._mute_getter = mute_getter or _get_master_mute
        self._mute_setter = mute_setter or _set_master_mute

    def execute(self, action: str = "", **kwargs: Any) -> ToolResult:
        """
        Execute system audio adjustment.

        Args:
            action: One of 'volume_up', 'volume_down', 'mute', or 'unmute'.
        """
        if platform.system() != "Windows":
            return ToolResult.fail(
                error="UNSUPPORTED_PLATFORM",
                message="System audio control is only supported on Windows.",
                error_code="UNSUPPORTED_PLATFORM",
            )

        if not action or not isinstance(action, str) or action not in ALLOWED_ACTIONS:
            allowed = ", ".join(sorted(ALLOWED_ACTIONS))
            return ToolResult.fail(
                error="INVALID_ACTION",
                message=f"Action '{action}' is not supported. Must be one of: {allowed}.",
                error_code="INVALID_ARGUMENTS",
            )

        try:
            if action in ("mute", "unmute"):
                target_mute = (action == "mute")
                currently_muted = self._mute_getter()
                if currently_muted == target_mute:
                    state_str = "muted" if currently_muted else "unmuted"
                    return ToolResult.ok(
                        message=f"System audio is already {state_str}.",
                        data={"action": action, "muted": currently_muted, "changed": False},
                    )
                self._mute_setter(target_mute)
                state_str = "Muted" if target_mute else "Unmuted"
                return ToolResult.ok(
                    message=f"{state_str} system audio.",
                    data={"action": action, "muted": target_mute, "changed": True},
                )

            # volume_up / volume_down
            vk_code = VOLUME_KEY_MAP[action]
            self._key_sender(vk_code)
            action_msg = "Increased system volume." if action == "volume_up" else "Decreased system volume."
            return ToolResult.ok(
                message=action_msg,
                data={"action": action, "vk_code": vk_code},
            )

        except Exception as exc:
            logger.error("Failed to execute audio action '%s' (%s)", action, type(exc).__name__)
            return ToolResult.fail(
                error=str(exc),
                message=f"Failed to execute audio action '{action}': {exc}",
                error_code="EXECUTION_ERROR",
            )


def register_system_audio_tools(registry: ToolRegistry) -> None:
    """Register system audio control tools with a ToolRegistry."""
    registry.register(SystemAudioTool(), overwrite=True)
