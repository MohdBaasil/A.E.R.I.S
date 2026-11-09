"""
Manual real-world read-only diagnostic test for AERIS System Information Tool.

NOTE: This script inspects live hardware and OS parameters of this computer.
It is strictly READ-ONLY and performs NO state or configuration changes.
"""

import sys
import os

# Ensure root AERIS directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.system import SystemInfoTool, register_system_tools
from tools.registry import ToolRegistry


def main():
    print("=" * 60)
    print("       AERIS SYSTEM INFORMATION TEST (LIVE READ-ONLY)")
    print("=" * 60)
    print("\n[NOTE] Reading live system metrics using SystemInfoTool...\n")

    registry = ToolRegistry()
    register_system_tools(registry)

    result = registry.execute("system_info")

    if not result.success:
        print(f"[ERROR] Failed to retrieve system information: {result.error}")
        return

    data = result.data
    cpu = data.get("cpu", {})
    mem = data.get("memory", {})
    disk = data.get("disk", {})
    battery = data.get("battery", {})
    os_info = data.get("os", {})

    print("-" * 60)
    # CPU
    if cpu.get("available"):
        print(f"CPU: {cpu.get('usage_percent')}% utilization (Logical: {cpu.get('logical_cores')}, Physical: {cpu.get('physical_cores')})")
    else:
        print(f"CPU: Unavailable ({cpu.get('reason')})")

    # RAM
    if mem.get("available"):
        print(f"RAM: {mem.get('used_gb')} GB used / {mem.get('total_gb')} GB total ({mem.get('usage_percent')}%) [Available: {mem.get('available_gb')} GB]")
    else:
        print(f"RAM: Unavailable ({mem.get('reason')})")

    # DISK
    if disk.get("available"):
        print(f"DISK: Drive {disk.get('path')} - {disk.get('used_gb')} GB used / {disk.get('total_gb')} GB total ({disk.get('usage_percent')}%) [Free: {disk.get('free_gb')} GB]")

    else:
        print(f"DISK: Unavailable ({disk.get('reason')})")

    # BATTERY
    if battery.get("available"):
        plug_status = "Plugged In (Charging/AC)" if battery.get("plugged_in") else "On Battery"
        print(f"BATTERY: {battery.get('percent')}% [{plug_status}]")
    else:
        print(f"BATTERY: {battery.get('reason')}")

    # OS
    if os_info.get("available"):
        print(f"OS: {os_info.get('system')} {os_info.get('release')} (Build {os_info.get('version')}) [{os_info.get('machine')}]")
        print(f"HOSTNAME: {os_info.get('hostname')}")
    else:
        print(f"OS: Unavailable ({os_info.get('reason')})")
        print("HOSTNAME: Unavailable")

    print("-" * 60)
    print("\nSummary Status: " + result.message)
    print("=" * 60)


if __name__ == "__main__":
    main()
