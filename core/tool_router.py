"""
AI Tool Router for AERIS.

Bridges the AI model's structured tool-call requests to the ToolRegistry.
Enforces permission levels, validates tool names and arguments, and ensures
no arbitrary code execution ever reaches the operating system.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools.base import ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry
from core.confirmation import ConfirmationHandler

logger = logging.getLogger("aeris.core.tool_router")


@dataclass
class ToolCall:
    """Structured representation of a single tool invocation request."""
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterResult:
    """
    Outcome of the tool router processing a model response.

    Attributes:
        is_tool_call: True if the model requested a tool.
        tool_calls: List of parsed ToolCall objects.
        tool_results: List of ToolResult objects from execution.
        text_response: Any non-tool text the model produced.
        error: Error description if routing failed.
    """
    is_tool_call: bool = False
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    text_response: str = ""
    error: Optional[str] = None


# Regex to find JSON tool call blocks in model output
_TOOL_CALL_PATTERN = re.compile(
    r"```tool_call\s*\n(.*?)\n\s*```",
    re.DOTALL,
)


class ToolRouter:
    """
    Routes structured tool requests from the AI model to the ToolRegistry.

    Security guarantees:
    - Only tools registered in the ToolRegistry can be executed.
    - CONFIRM and DANGEROUS tools require confirmation from a ConfirmationHandler.
    - No shell commands, code execution, or dynamic imports are ever performed.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        confirmation_handler: Optional[ConfirmationHandler] = None,
    ) -> None:
        self._registry = registry
        self._confirmation = confirmation_handler

    def set_confirmation_handler(self, handler: ConfirmationHandler) -> None:
        """Update the confirmation handler (e.g. when switching text/voice mode)."""
        self._confirmation = handler

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return registered tool schemas for providers with native tool support."""
        return self._registry.get_schemas()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_tool_calls(self, model_output: str) -> List[ToolCall]:
        """
        Extract structured tool call(s) from a model response string.

        Expected format embedded in the model output:

            ```tool_call
            {"tool": "open_application", "arguments": {"app_name": "notepad"}}
            ```

        Returns a list of ToolCall objects. Returns empty list if no
        tool calls are found.
        """
        calls: List[ToolCall] = []
        matches = _TOOL_CALL_PATTERN.findall(model_output)

        for raw_json in matches:
            try:
                parsed = json.loads(raw_json.strip())
            except json.JSONDecodeError as exc:
                logger.warning("Malformed tool call JSON: %s", exc)
                continue

            if not isinstance(parsed, dict):
                logger.warning("Tool call payload is not a JSON object")
                continue

            tool_name = parsed.get("tool", "")
            arguments = parsed.get("arguments", {})

            if not tool_name or not isinstance(tool_name, str):
                logger.warning("Tool call payload is missing a valid tool name")
                continue

            if not isinstance(arguments, dict):
                arguments = {}

            calls.append(ToolCall(tool=tool_name.strip(), arguments=arguments))

        return calls

    def strip_tool_blocks(self, model_output: str) -> str:
        """Remove tool_call code blocks from model output, leaving prose."""
        return _TOOL_CALL_PATTERN.sub("", model_output).strip()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_tool_call(self, call: ToolCall) -> ToolResult:
        """
        Validate and execute a single ToolCall through the registry.
        """
        tool_name = call.tool

        # Security: reject if tool is not registered
        if not self._registry.has(tool_name):
            logger.warning("Model requested an unregistered tool")
            return ToolResult.fail(
                error="UNKNOWN_TOOL",
                message=f"Tool '{tool_name}' is not available in AERIS.",
            )

        tool = self._registry.get(tool_name)
        # Validate before asking for approval. Invalid calls cannot trigger a
        # permission prompt or reach the underlying tool.
        invalid = self._registry.validate_arguments(tool_name, call.arguments)
        if invalid is not None:
            logger.warning("Rejected invalid arguments for tool '%s'", tool_name)
            return invalid
        permission = tool.permission_level

        # Permission gate
        if permission in (PermissionLevel.CONFIRM, PermissionLevel.DANGEROUS):
            if self._confirmation is None:
                logger.warning(
                    "Tool '%s' requires %s but no confirmation handler is set. Denying.",
                    tool_name, permission.value,
                )
                return ToolResult.fail(
                    error="CONFIRMATION_UNAVAILABLE",
                    message=f"Tool '{tool_name}' requires {permission.value} permission but confirmation is not available.",
                )

            # Confirmation must describe the actual validated target/action so
            # the user can make an informed approval decision. This is not logged.
            description = f"{tool.description} (arguments: {call.arguments})"
            approved = self._confirmation.request_confirmation(
                tool_name=tool_name,
                description=description,
                permission_level=permission.value,
            )
            if not approved:
                logger.info("User denied execution of tool '%s'", tool_name)
                return ToolResult.fail(
                    error="PERMISSION_DENIED",
                    message=f"You declined to execute '{tool_name}'.",
                )

        # Execute through registry (safe wrapper)
        logger.info("Dispatching validated tool execution: %s", tool_name)
        return self._registry.execute(tool_name, **call.arguments)

    # ------------------------------------------------------------------
    # Full routing pipeline
    # ------------------------------------------------------------------

    def process_model_response(self, model_output: str) -> RouterResult:
        """
        Process a complete model response: detect tool calls, execute them,
        and return structured results.

        Returns a RouterResult containing:
        - Whether tool calls were found
        - The tool results
        - Any remaining text response
        """
        tool_calls = self.parse_tool_calls(model_output)
        remaining_text = self.strip_tool_blocks(model_output)

        if not tool_calls:
            # Pure conversational response, no tools
            return RouterResult(
                is_tool_call=False,
                text_response=remaining_text or model_output,
            )

        # Execute each tool call sequentially
        results: List[ToolResult] = []
        for call in tool_calls:
            result = self.execute_tool_call(call)
            results.append(result)

        return RouterResult(
            is_tool_call=True,
            tool_calls=tool_calls,
            tool_results=results,
            text_response=remaining_text,
        )

    # ------------------------------------------------------------------
    # Tool schema export (for system prompt construction)
    # ------------------------------------------------------------------

    def get_tool_definitions_text(self) -> str:
        """
        Build a text block describing all available tools for the system prompt.
        """
        schemas = self._registry.get_schemas()
        if not schemas:
            return "No tools are currently available."

        lines = ["You have access to the following tools:\n"]
        for schema in schemas:
            name = schema["name"]
            desc = schema["description"]
            params = schema.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            perm = schema.get("permission_level", "SAFE")

            lines.append(f"- **{name}** [{perm}]: {desc}")

            properties = params.get("properties", {})
            required = params.get("required", [])
            if isinstance(properties, dict) and properties:
                for pname, pinfo in properties.items():
                    req_marker = " (required)" if pname in required else " (optional)"
                    pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
                    lines.append(f"    - {pname}{req_marker}: {pdesc}")

        lines.append("")
        return "\n".join(lines)
