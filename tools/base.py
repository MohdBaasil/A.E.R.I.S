"""
Base abstractions for the AERIS Tool System.

Provides the foundational Tool abstract class and the structured ToolResult model.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tools.permissions import PermissionLevel

logger = logging.getLogger("aeris.tools")

_SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|secret|token|password|auth|bearer)\s*[:=]\s*["\']?([^"\'\s&]+)["\']?'),
    re.compile(r'AIza[0-9A-Za-z-_]{20,}'),
]


def sanitize_value(val: Any) -> Any:
    """Recursively scrub known sensitive patterns from values."""
    if isinstance(val, str):
        cleaned = val
        for pat in _SECRET_PATTERNS:
            cleaned = pat.sub(r'\1: [REDACTED]' if pat.groups else '[REDACTED]', cleaned)
        return cleaned
    if isinstance(val, dict):
        return {k: sanitize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [sanitize_value(item) for item in val]
    return val


@dataclass
class ToolResult:
    """
    Structured outcome returned by tool execution.
    
    Attributes:
        success: True if the tool executed without error, False otherwise.
        message: Human-readable summary of the outcome.
        data: Optional structured payload returned by the tool.
        error: Error message or description if execution failed.
    """
    success: bool
    message: str = ""
    data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None

    @classmethod
    def ok(cls, message: str = "", data: Any = None) -> "ToolResult":
        """Create a successful ToolResult."""
        return cls(success=True, message=message, data=data, error=None)

    @classmethod
    def fail(cls, error: str, message: str = "", data: Any = None, error_code: Optional[str] = None) -> "ToolResult":
        """Create a failed ToolResult."""
        inferred = error_code or (error if re.fullmatch(r"[A-Z][A-Z0-9_]*", str(error)) else "EXECUTION_ERROR")
        return cls(success=False, message=message or error, data=data, error=error, error_code=inferred)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into a serializable dictionary without exposing secrets."""
        return {
            "success": self.success,
            "message": sanitize_value(self.message),
            "data": sanitize_value(self.data),
            "error": sanitize_value(self.error),
            "error_code": self.error_code,
        }


class Tool(ABC):
    """
    Abstract base class for all AERIS tools.
    
    Subclasses must define:
        name: Unique identifier for the tool.
        description: Clear explanation of what the tool does for the AI.
        parameters: Schema defining required/optional arguments.
        permission_level: Safety rating (SAFE, CONFIRM, DANGEROUS).
        execute(**kwargs): Concrete execution logic returning a ToolResult.
    """
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    permission_level: PermissionLevel = PermissionLevel.SAFE

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        permission_level: Optional[PermissionLevel] = None,
    ) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        if permission_level is not None:
            self.permission_level = permission_level

        if not self.name:
            raise ValueError(f"Tool class {self.__class__.__name__} must define a non-empty 'name'.")

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool action with the provided keyword arguments.
        
        Must return a ToolResult instance.
        """
        raise NotImplementedError

    def run(self, **kwargs: Any) -> ToolResult:
        """
        Safe execution wrapper that catches uncaught exceptions and logs activity.
        """
        started = time.perf_counter()
        try:
            result = self.execute(**kwargs)
            if not isinstance(result, ToolResult):
                result = ToolResult.fail(
                    error=f"Tool '{self.name}' did not return a ToolResult.",
                    message="Internal tool execution error.",
                )
        except Exception as exc:
            # Exception strings can contain user data or credentials; keep logs
            # diagnostic without writing the exception text or traceback.
            logger.error("Uncaught exception during tool '%s' (%s)", self.name, type(exc).__name__)
            result = ToolResult.fail(
                error=str(exc),
                message=f"Execution of tool '{self.name}' failed due to an unhandled exception.",
                error_code="EXECUTION_ERROR",
            )
        logger.info(
            "Tool execution complete: tool=%s success=%s duration_ms=%.2f error_code=%s",
            self.name, result.success, (time.perf_counter() - started) * 1000,
            result.error_code if not result.success else None,
        )
        return result

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the standardized tool declaration schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters or {},
            "permission_level": self.permission_level.value,
        }
