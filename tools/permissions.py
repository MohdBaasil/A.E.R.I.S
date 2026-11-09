"""
Permission model for the AERIS Tool System.

Defines safety levels for tool execution to support user confirmation
and security boundaries in subsequent phases.
"""

from enum import Enum


class PermissionLevel(str, Enum):
    """
    Permission levels governing tool execution authority.
    
    SAFE:
        Read-only or non-destructive actions (e.g. system info, search, read file).
        Can be executed automatically.
        
    CONFIRM:
        Modifying actions that change system/app state (e.g. create folder, open app, move file).
        May require user confirmation in interactive modes.
        
    DANGEROUS:
        Destructive or high-risk actions (e.g. delete file, terminate process).
        Requires explicit user authorization.
    """
    SAFE = "SAFE"
    CONFIRM = "CONFIRM"
    DANGEROUS = "DANGEROUS"
