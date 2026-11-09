"""
System Information Tool for AERIS.

Provides safe, read-only diagnostics for CPU, Memory, Disk, Battery, and OS.
Does not perform any state-modifying operations.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
from typing import Any, Dict, Optional

from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry

logger = logging.getLogger("aeris.tools.system")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.warning("psutil is not installed. Some detailed hardware statistics may be limited.")


class SystemInfoTool(Tool):
    """
    Read-only tool that gathers current system diagnostic and resource information.
    """
    name = "system_info"
    description = (
        "Retrieves read-only hardware and operating system metrics (CPU, RAM, Disk, Battery, OS)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "disk_path": {
                "type": "string",
                "description": "Optional drive path to check disk usage (default is 'C:\\\\' on Windows or '/').",
                "default": "C:\\" if platform.system() == "Windows" else "/",
            }
        },
        "required": [],
    }
    permission_level = PermissionLevel.SAFE

    def __init__(self) -> None:
        super().__init__()

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Gather CPU metrics."""
        try:
            if _PSUTIL_AVAILABLE:
                # Interval=0.1 allows a quick snapshot measurement
                usage = psutil.cpu_percent(interval=0.1)
                logical = psutil.cpu_count(logical=True)
                physical = psutil.cpu_count(logical=False)
                return {
                    "available": True,
                    "usage_percent": usage,
                    "logical_cores": logical,
                    "physical_cores": physical,
                }
            else:
                return {
                    "available": True,
                    "usage_percent": None,
                    "logical_cores": os.cpu_count(),
                    "physical_cores": None,
                    "note": "psutil unavailable",
                }
        except Exception as exc:
            logger.error("Failed to gather CPU info: %s", exc)
            return {
                "available": False,
                "error": str(exc),
                "reason": "CPU information unavailable",
            }

    def _get_memory_info(self) -> Dict[str, Any]:
        """Gather RAM memory metrics."""
        try:
            if _PSUTIL_AVAILABLE:
                vmem = psutil.virtual_memory()
                return {
                    "available": True,
                    "total_bytes": vmem.total,
                    "used_bytes": vmem.used,
                    "available_bytes": vmem.available,
                    "total_gb": round(vmem.total / (1024 ** 3), 2),
                    "used_gb": round(vmem.used / (1024 ** 3), 2),
                    "available_gb": round(vmem.available / (1024 ** 3), 2),
                    "usage_percent": vmem.percent,
                }
            else:
                return {
                    "available": False,
                    "reason": "Memory information requires psutil",
                }
        except Exception as exc:
            logger.error("Failed to gather memory info: %s", exc)
            return {
                "available": False,
                "error": str(exc),
                "reason": "Memory information unavailable",
            }

    def _get_disk_info(self, disk_path: Optional[str] = None) -> Dict[str, Any]:
        """Gather storage disk metrics for the specified drive path."""
        target_path = disk_path or ("C:\\" if platform.system() == "Windows" else "/")
        try:
            if _PSUTIL_AVAILABLE:
                disk = psutil.disk_usage(target_path)
                return {
                    "available": True,
                    "path": target_path,
                    "total_bytes": disk.total,
                    "used_bytes": disk.used,
                    "free_bytes": disk.free,
                    "total_gb": round(disk.total / (1024 ** 3), 2),
                    "used_gb": round(disk.used / (1024 ** 3), 2),
                    "free_gb": round(disk.free / (1024 ** 3), 2),
                    "usage_percent": disk.percent,
                }
            else:
                import shutil
                usage = shutil.disk_usage(target_path)
                total_gb = round(usage.total / (1024 ** 3), 2)
                used_gb = round(usage.used / (1024 ** 3), 2)
                free_gb = round(usage.free / (1024 ** 3), 2)
                usage_percent = round((usage.used / usage.total) * 100, 1) if usage.total > 0 else 0.0
                return {
                    "available": True,
                    "path": target_path,
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                    "total_gb": total_gb,
                    "used_gb": used_gb,
                    "free_gb": free_gb,
                    "usage_percent": usage_percent,
                }
        except Exception as exc:
            logger.error("Failed to gather disk info for path '%s': %s", target_path, exc)
            return {
                "available": False,
                "path": target_path,
                "error": str(exc),
                "reason": f"Disk information unavailable for '{target_path}'",
            }

    def _get_battery_info(self) -> Dict[str, Any]:
        """Gather battery status if available."""
        try:
            if _PSUTIL_AVAILABLE:
                battery = psutil.sensors_battery()
                if battery is None:
                    return {
                        "available": False,
                        "percent": None,
                        "plugged_in": None,
                        "reason": "No battery detected (Desktop PC or unsupported hardware)",
                    }
                return {
                    "available": True,
                    "percent": round(battery.percent, 1),
                    "plugged_in": battery.power_plugged,
                    "secsleft": battery.secsleft if battery.secsleft != getattr(psutil, "POWER_TIME_UNLIMITED", -1) else "unlimited",
                }
            else:
                return {
                    "available": False,
                    "percent": None,
                    "plugged_in": None,
                    "reason": "Battery diagnostics requires psutil",
                }
        except Exception as exc:
            logger.error("Failed to gather battery info: %s", exc)
            return {
                "available": False,
                "percent": None,
                "plugged_in": None,
                "error": str(exc),
                "reason": "Battery information unavailable",
            }

    def _get_os_info(self) -> Dict[str, Any]:
        """Gather operating system and machine identity."""
        try:
            return {
                "available": True,
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "hostname": socket.gethostname(),
            }
        except Exception as exc:
            logger.error("Failed to gather OS info: %s", exc)
            return {
                "available": False,
                "error": str(exc),
                "reason": "OS information unavailable",
            }

    def execute(self, disk_path: Optional[str] = None, **kwargs: Any) -> ToolResult:
        """
        Execute system information retrieval.
        
        Args:
            disk_path: Optional path for disk usage inspection.
        """
        cpu_data = self._get_cpu_info()
        mem_data = self._get_memory_info()
        disk_data = self._get_disk_info(disk_path)
        battery_data = self._get_battery_info()
        os_data = self._get_os_info()

        data = {
            "cpu": cpu_data,
            "memory": mem_data,
            "disk": disk_data,
            "battery": battery_data,
            "os": os_data,
        }

        # Build a concise summary message
        cpu_str = f"CPU: {cpu_data.get('usage_percent')}%" if cpu_data.get("available") and cpu_data.get("usage_percent") is not None else "CPU: Active"
        mem_str = f"RAM: {mem_data.get('used_gb')}/{mem_data.get('total_gb')} GB ({mem_data.get('usage_percent')}%)" if mem_data.get("available") else "RAM: N/A"
        
        message = f"System information retrieved. {cpu_str}, {mem_str}."

        return ToolResult.ok(message=message, data=data)


def register_system_tools(registry: ToolRegistry) -> None:
    """
    Register all system diagnostic tools with a ToolRegistry.
    """
    registry.register(SystemInfoTool(), overwrite=True)
