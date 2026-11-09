"""
Manual real-world test for AERIS Application Control.

WARNING: Running this script will launch a REAL Windows application (Notepad).
"""

import sys
import os

# Ensure root AERIS directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.applications import OpenApplicationTool, register_application_tools
from tools.registry import ToolRegistry



def main():
    print("=" * 60)
    print("   AERIS MANUAL APPLICATION LAUNCH TEST")
    print("=" * 60)
    print("\n[NOTE] This test will attempt to launch Notepad on your Windows system.")
    print("\n[1] Initializing ToolRegistry with OpenApplicationTool...")

    registry = ToolRegistry()
    register_application_tools(registry)

    print("    - Tool registered: open_application")

    # Test 1: Allowlisted application launch
    print("\n[2] Executing 'open_application' with app_name='notepad'...")
    result = registry.execute("open_application", app_name="notepad")

    print(f"    - Success : {result.success}")
    print(f"    - Message : {result.message}")
    print(f"    - Data    : {result.data}")
    print(f"    - Error   : {result.error}")

    # Test 2: Security check - unknown application
    print("\n[3] Security Test: Executing with unknown app_name='forbidden_command'...")
    sec_result = registry.execute("open_application", app_name="forbidden_command")

    print(f"    - Success : {sec_result.success} (Expected: False)")
    print(f"    - Error   : {sec_result.error} (Expected: UNKNOWN_APPLICATION)")
    print(f"    - Message : {sec_result.message}")

    print("\n" + "=" * 60)
    if result.success and not sec_result.success:
        print("   MANUAL APPLICATION CONTROL TEST: SUCCESS")
    else:
        print("   MANUAL APPLICATION CONTROL TEST: FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
