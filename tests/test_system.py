from unittest.mock import patch, MagicMock
from collections import namedtuple
import pytest

from tools.system import SystemInfoTool, register_system_tools
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry


# ==========================================
# Tool Metadata & Registration Tests
# ==========================================

def test_system_info_tool_metadata():
    tool = SystemInfoTool()
    assert tool.name == "system_info"
    assert tool.permission_level == PermissionLevel.SAFE
    assert "disk_path" in tool.parameters["properties"]

    schema = tool.get_schema()
    assert schema["name"] == "system_info"
    assert schema["permission_level"] == "SAFE"


def test_registry_integration():
    registry = ToolRegistry()
    register_system_tools(registry)

    assert registry.has("system_info") is True
    tool = registry.get("system_info")
    assert isinstance(tool, SystemInfoTool)


# ==========================================
# Deterministic Mock Execution Tests
# ==========================================

@patch("psutil.cpu_percent", return_value=15.5)
@patch("psutil.cpu_count")
@patch("psutil.virtual_memory")
@patch("psutil.disk_usage")
@patch("psutil.sensors_battery")
@patch("platform.system", return_value="Windows")
@patch("platform.release", return_value="11")
@patch("platform.version", return_value="10.0.22631")
@patch("platform.machine", return_value="AMD64")
@patch("socket.gethostname", return_value="AERIS-PC")
def test_system_info_successful_execution_with_battery(
    mock_host, mock_mach, mock_ver, mock_rel, mock_sys,
    mock_battery, mock_disk, mock_vmem, mock_cpu_count, mock_cpu_pct
):
    # Mock CPU counts: logical=16, physical=8
    mock_cpu_count.side_effect = lambda logical: 16 if logical else 8

    # Mock virtual memory (16 GB total, 8 GB used)
    VMem = namedtuple("VMem", ["total", "used", "available", "percent"])
    mock_vmem.return_value = VMem(
        total=16 * 1024**3,
        used=8 * 1024**3,
        available=8 * 1024**3,
        percent=50.0,
    )

    # Mock Disk (512 GB total, 256 GB used, 256 GB free)
    DiskUsage = namedtuple("DiskUsage", ["total", "used", "free", "percent"])
    mock_disk.return_value = DiskUsage(
        total=512 * 1024**3,
        used=256 * 1024**3,
        free=256 * 1024**3,
        percent=50.0,
    )

    # Mock Battery (85%, plugged in)
    Battery = namedtuple("Battery", ["percent", "secsleft", "power_plugged"])
    mock_battery.return_value = Battery(
        percent=85.0,
        secsleft=7200,
        power_plugged=True,
    )

    tool = SystemInfoTool()
    result = tool.run()

    assert result.success is True
    assert "System information retrieved" in result.message
    data = result.data

    # Verify CPU
    assert data["cpu"]["available"] is True
    assert data["cpu"]["usage_percent"] == 15.5
    assert data["cpu"]["logical_cores"] == 16
    assert data["cpu"]["physical_cores"] == 8

    # Verify Memory
    assert data["memory"]["available"] is True
    assert data["memory"]["total_gb"] == 16.0
    assert data["memory"]["used_gb"] == 8.0
    assert data["memory"]["usage_percent"] == 50.0

    # Verify Disk
    assert data["disk"]["available"] is True
    assert data["disk"]["total_gb"] == 512.0
    assert data["disk"]["used_gb"] == 256.0
    assert data["disk"]["free_gb"] == 256.0
    assert data["disk"]["usage_percent"] == 50.0

    # Verify Battery
    assert data["battery"]["available"] is True
    assert data["battery"]["percent"] == 85.0
    assert data["battery"]["plugged_in"] is True

    # Verify OS
    assert data["os"]["system"] == "Windows"
    assert data["os"]["release"] == "11"
    assert data["os"]["hostname"] == "AERIS-PC"


@patch("psutil.cpu_percent", return_value=5.0)
@patch("psutil.cpu_count", return_value=8)
@patch("psutil.virtual_memory")
@patch("psutil.disk_usage")
@patch("psutil.sensors_battery", return_value=None)  # Desktop PC / No Battery
def test_system_info_no_battery_graceful(mock_battery, mock_disk, mock_vmem, mock_cpu_count, mock_cpu_pct):
    VMem = namedtuple("VMem", ["total", "used", "available", "percent"])
    mock_vmem.return_value = VMem(total=8 * 1024**3, used=4 * 1024**3, available=4 * 1024**3, percent=50.0)
    DiskUsage = namedtuple("DiskUsage", ["total", "used", "free", "percent"])
    mock_disk.return_value = DiskUsage(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3, percent=50.0)

    tool = SystemInfoTool()
    result = tool.run()

    assert result.success is True
    assert result.data["battery"]["available"] is False
    assert result.data["battery"]["percent"] is None
    assert "No battery detected" in result.data["battery"]["reason"]


# ==========================================
# Robustness & Partial Failure Tests
# ==========================================

@patch("psutil.cpu_percent", side_effect=RuntimeError("WMI error"))
@patch("psutil.virtual_memory", side_effect=OSError("Memory query failed"))
@patch("psutil.disk_usage", side_effect=FileNotFoundError("Invalid drive"))
@patch("psutil.sensors_battery", side_effect=Exception("Battery query crash"))
def test_system_info_handles_component_exceptions(mock_bat, mock_disk, mock_vmem, mock_cpu):
    tool = SystemInfoTool()
    # Tool must NOT crash; it should return available=False for failed components
    result = tool.run()

    assert result.success is True
    assert result.data["cpu"]["available"] is False
    assert "CPU information unavailable" in result.data["cpu"]["reason"]

    assert result.data["memory"]["available"] is False
    assert "Memory information unavailable" in result.data["memory"]["reason"]

    assert result.data["disk"]["available"] is False
    assert "Disk information unavailable" in result.data["disk"]["reason"]

    assert result.data["battery"]["available"] is False
    assert "Battery information unavailable" in result.data["battery"]["reason"]

    # OS should still be available
    assert result.data["os"]["available"] is True
