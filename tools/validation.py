"""Small JSON-schema validator for the subset used by registered AERIS tools."""

from __future__ import annotations

from typing import Any


class ArgumentValidationError(ValueError):
    pass


def validate_arguments(schema: dict[str, Any], arguments: Any) -> None:
    """Validate tool arguments before permission prompts or execution."""
    if not isinstance(arguments, dict):
        raise ArgumentValidationError("arguments must be a JSON object")
    _validate_value(schema, arguments, "arguments", reject_extra=True)


def _validate_value(schema: Any, value: Any, path: str, reject_extra: bool = False) -> None:
    if not isinstance(schema, dict):
        raise ArgumentValidationError(f"invalid schema for {path}")

    expected = schema.get("type")
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if expected in checks and not checks[expected](value):
        raise ArgumentValidationError(f"{path} must be {expected}")
    if expected is not None and expected not in checks:
        raise ArgumentValidationError(f"unsupported schema type {expected!r} for {path}")

    if "enum" in schema and value not in schema["enum"]:
        raise ArgumentValidationError(f"{path} must be one of the allowed values")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ArgumentValidationError(f"invalid object schema for {path}")
        missing = [key for key in required if key not in value]
        if missing:
            raise ArgumentValidationError(f"{path} is missing required field(s): {', '.join(map(str, missing))}")
        extra = set(value) - set(properties)
        if extra and (reject_extra or schema.get("additionalProperties") is False):
            raise ArgumentValidationError(f"{path} contains unexpected field(s): {', '.join(sorted(map(str, extra)))}")
        for key, item in value.items():
            if key in properties:
                _validate_value(properties[key], item, f"{path}.{key}", reject_extra=True)
    elif isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate_value(schema["items"], item, f"{path}[{index}]", reject_extra=True)
