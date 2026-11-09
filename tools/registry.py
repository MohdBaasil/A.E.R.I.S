"""
Tool Registry for AERIS.

Manages tool registration, lookup, inspection, and execution dispatching.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tools.base import Tool, ToolResult
from tools.validation import ArgumentValidationError, validate_arguments

logger = logging.getLogger("aeris.tools.registry")


class ToolRegistryError(Exception):
    """Base exception for tool registry operations."""


class ToolNotFoundError(ToolRegistryError):
    """Raised or returned when attempting to access an unregistered tool."""


class ToolRegistry:
    """
    Central catalog of available AERIS tools.
    
    Provides thread-safe registration, enumeration, lookup, and safe execution.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool, overwrite: bool = False) -> None:
        """
        Register a tool instance in the registry.
        
        Args:
            tool: Tool instance to register.
            overwrite: If True, replace an existing tool with the same name.
            
        Raises:
            TypeError: If tool is not an instance of Tool.
            ValueError: If a tool with the same name is already registered and overwrite is False.
        """
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected Tool instance, got '{type(tool).__name__}'")

        name = tool.name
        if name in self._tools and not overwrite:
            raise ValueError(f"Tool '{name}' is already registered in ToolRegistry.")

        self._tools[name] = tool
        logger.info("Registered tool '%s' (permission: %s)", name, tool.permission_level.value)

    def unregister(self, name: str) -> Optional[Tool]:
        """
        Remove a tool from the registry by name.
        
        Returns the removed Tool instance, or None if it wasn't registered.
        """
        tool = self._tools.pop(name, None)
        if tool:
            logger.info("Unregistered tool '%s'", name)
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """
        Retrieve a tool by name. Returns None if not found.
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """
        Check if a tool with the given name is registered.
        """
        return name in self._tools

    def list_tools(self) -> List[Tool]:
        """
        Return a list of all registered Tool instances.
        """
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """
        Return a list of all registered tool names.
        """
        return list(self._tools.keys())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Return declaration schemas for all registered tools.
        """
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, /, **kwargs: Any) -> ToolResult:
        """
        Execute a registered tool by name with arguments.
        
        If the tool does not exist, returns a failed ToolResult.
        """
        tool = self.get(tool_name)
        if tool is None:
            logger.warning("Attempted execution of an unknown tool")
            return ToolResult.fail(
                error=f"Tool '{tool_name}' not found.",
                message=f"Requested tool '{tool_name}' is not registered in the system.",
            )

        invalid = self.validate_arguments(tool_name, kwargs)
        if invalid is not None:
            return invalid

        return tool.run(**kwargs)

    def validate_arguments(self, tool_name: str, arguments: Any) -> Optional[ToolResult]:
        """Return a failed result when arguments do not match a registered schema."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult.fail(error="UNKNOWN_TOOL", message=f"Tool '{tool_name}' is not registered.")
        try:
            validate_arguments(tool.parameters or {"type": "object", "properties": {}}, arguments)
        except ArgumentValidationError as exc:
            return ToolResult.fail(
                error="INVALID_ARGUMENTS",
                message=f"Invalid arguments for '{tool_name}': {exc}.",
                error_code="INVALID_ARGUMENTS",
            )
        return None

    def clear(self) -> None:
        """Remove all registered tools from the registry."""
        self._tools.clear()
