"""
Manual test for AERIS File Management Tools.

IMPORTANT: This test operates ONLY inside an isolated temporary directory.
It does NOT modify any user or personal files.
"""

import os
import sys
import tempfile
import pathlib

# Ensure root AERIS directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.registry import ToolRegistry
from tools.files import register_file_tools


def main():
    print("=" * 60)
    print("      AERIS FILE MANAGEMENT MANUAL TEST (SANDBOX)")
    print("=" * 60)
    print("\n[NOTE] This test operates strictly inside a temporary directory.")
    print("[NOTE] No personal or system files will be modified.\n")

    # Step 1: Initialize registry and register tools
    registry = ToolRegistry()
    register_file_tools(registry)

    # Step 1 & 2: Create temporary workspace
    with tempfile.TemporaryDirectory(prefix="aeris_file_test_") as temp_dir_str:
        temp_dir = pathlib.Path(temp_dir_str)
        print(f"[1] Created isolated temporary directory:\n    {temp_dir}\n")

        # Create initial test files
        sample_file_1 = temp_dir / "document_1.txt"
        sample_file_1.write_text("AERIS test file content 1", encoding="utf-8")

        sample_file_2 = temp_dir / "notes_project.md"
        sample_file_2.write_text("# Project Notes\nSample content", encoding="utf-8")

        print("[2] Created initial test files: 'document_1.txt', 'notes_project.md'")

        # Step 3: List files
        print("\n[3] Testing 'list_files'...")
        res_list = registry.execute("list_files", path=str(temp_dir))
        print(f"    - Success : {res_list.success}")
        print(f"    - Message : {res_list.message}")
        print(f"    - Items   : {[i['name'] for i in res_list.data.get('items', [])]}")

        # Step 4: Search files
        print("\n[4] Testing 'search_files' for query 'project'...")
        res_search = registry.execute("search_files", directory=str(temp_dir), query="project")
        print(f"    - Success : {res_search.success}")
        print(f"    - Matches : {[m['name'] for m in res_search.data.get('matches', [])]}")

        # Step 5: Get file info
        print("\n[5] Testing 'get_file_info' on 'document_1.txt'...")
        res_info = registry.execute("get_file_info", path=str(sample_file_1))
        print(f"    - Success : {res_info.success}")
        print(f"    - Name    : {res_info.data.get('name')}")
        print(f"    - Size    : {res_info.data.get('size_bytes')} bytes")
        print(f"    - Type    : {res_info.data.get('type')}")
        print(f"    - ModTime : {res_info.data.get('modified_time')}")

        # Step 6: Create subfolder
        subfolder_path = temp_dir / "Archive"
        print(f"\n[6] Testing 'create_folder' -> '{subfolder_path.name}'...")
        res_folder = registry.execute("create_folder", path=str(subfolder_path))
        print(f"    - Success : {res_folder.success}")
        print(f"    - Message : {res_folder.message}")

        # Step 7: Rename file
        print("\n[7] Testing 'rename_file' ('document_1.txt' -> 'document_renamed.txt')...")
        res_rename = registry.execute("rename_file", path=str(sample_file_1), new_name="document_renamed.txt")
        print(f"    - Success : {res_rename.success}")
        print(f"    - Message : {res_rename.message}")
        renamed_file = temp_dir / "document_renamed.txt"

        # Step 8: Move file into subfolder
        print(f"\n[8] Testing 'move_file' ('document_renamed.txt' -> '{subfolder_path.name}')...")
        res_move = registry.execute("move_file", source=str(renamed_file), destination=str(subfolder_path))
        print(f"    - Success : {res_move.success}")
        print(f"    - Message : {res_move.message}")

        # Step 9: Delete a file
        file_to_delete = temp_dir / "notes_project.md"
        print(f"\n[9] Testing 'delete_file' on '{file_to_delete.name}'...")
        res_del = registry.execute("delete_file", path=str(file_to_delete))
        print(f"    - Success : {res_del.success}")
        print(f"    - Message : {res_del.message}")

        # Security check: Wildcard rejection
        print("\n[10] Testing Security: Attempting wildcard deletion ('*.txt')...")
        res_sec = registry.execute("delete_file", path="*.txt")
        print(f"    - Blocked : {not res_sec.success} (Error: {res_sec.error})")

        print("\n[11] Cleaning up temporary test sandbox automatically...")

    print("=" * 60)
    print("   AERIS FILE MANAGEMENT MANUAL TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
