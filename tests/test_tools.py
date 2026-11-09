import pytest
from typing import Any, Dict

from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry


class DummyEchoTool(Tool):
    name = "dummy_echo"
    description = "Echoes the input message"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo"}
        },
        "required": ["text"],
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, text: str = "", **kwargs: Any) -> ToolResult:
        if not text:
            return ToolResult.fail("Text parameter is empty.", message="Failed to echo empty text.")
        return ToolResult.ok(message=f"Echo: {text}", data={"echoed": text})


class FailingTool(Tool):
    name = "failing_tool"
    description = "Tool that raises an unexpected exception"
    permission_level = PermissionLevel.DANGEROUS

    def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("Unexpected internal crash in tool logic")


class BadReturnTool(Tool):
    name = "bad_return_tool"
    description = "Tool that returns a non-ToolResult object"

    def execute(self, **kwargs: Any) -> Any:
        return "Not a ToolResult"


# ==========================================
# ToolResult Tests
# ==========================================

def test_tool_result_ok():
    result = ToolResult.ok(message="Operation succeeded", data={"count": 42})
    assert result.success is True
    assert result.message == "Operation succeeded"
    assert result.data == {"count": 42}
    assert result.error is None

    as_dict = result.to_dict()
    assert as_dict["success"] is True
    assert as_dict["data"]["count"] == 42
    assert as_dict["error"] is None


def test_tool_result_fail():
    result = ToolResult.fail(error="File not found", message="Unable to open file")
    assert result.success is False
    assert result.error == "File not found"
    assert result.message == "Unable to open file"
    assert result.data is None

    as_dict = result.to_dict()
    assert as_dict["success"] is False
    assert as_dict["error"] == "File not found"


# ==========================================
# Tool Base & Metadata Tests
# ==========================================

def test_tool_metadata_and_schema():
    tool = DummyEchoTool()
    assert tool.name == "dummy_echo"
    assert tool.description == "Echoes the input message"
    assert tool.permission_level == PermissionLevel.SAFE
    assert "text" in tool.parameters["properties"]

    schema = tool.get_schema()
    assert schema["name"] == "dummy_echo"
    assert schema["description"] == "Echoes the input message"
    assert schema["permission_level"] == "SAFE"
    assert "text" in schema["parameters"]["properties"]


def test_tool_instantiation_without_name_raises_error():
    class UnnamedTool(Tool):
        def execute(self, **kwargs: Any) -> ToolResult:
            return ToolResult.ok()

    with pytest.raises(ValueError, match="must define a non-empty 'name'"):
        UnnamedTool()



def test_tool_custom_init_override():
    class CustomTool(Tool):
        def execute(self, **kwargs: Any) -> ToolResult:
            return ToolResult.ok("done")

    tool = CustomTool(
        name="custom_name",
        description="custom desc",
        permission_level=PermissionLevel.CONFIRM,
        parameters={"type": "object"},
    )
    assert tool.name == "custom_name"
    assert tool.description == "custom desc"
    assert tool.permission_level == PermissionLevel.CONFIRM


# ==========================================
# Execution & Safety Wrapper Tests
# ==========================================

def test_tool_successful_execution():
    tool = DummyEchoTool()
    result = tool.run(text="Hello AERIS")
    assert result.success is True
    assert result.message == "Echo: Hello AERIS"
    assert result.data == {"echoed": "Hello AERIS"}


def test_tool_failed_execution_logic():
    tool = DummyEchoTool()
    result = tool.run(text="")
    assert result.success is False
    assert result.error == "Text parameter is empty."


def test_tool_uncaught_exception_handling():
    tool = FailingTool()
    result = tool.run()
    assert result.success is False
    assert "Unexpected internal crash in tool logic" in result.error
    assert "failed due to an unhandled exception" in result.message


def test_tool_invalid_return_type_handling():
    tool = BadReturnTool()
    result = tool.run()
    assert result.success is False
    assert "did not return a ToolResult" in result.error


# ==========================================
# ToolRegistry Tests
# ==========================================

def test_registry_registration_and_lookup():
    registry = ToolRegistry()
    tool = DummyEchoTool()

    assert registry.has("dummy_echo") is False
    registry.register(tool)
    assert registry.has("dummy_echo") is True
    assert registry.get("dummy_echo") is tool
    assert registry.list_names() == ["dummy_echo"]
    assert registry.list_tools() == [tool]


def test_registry_duplicate_registration():
    registry = ToolRegistry()
    tool1 = DummyEchoTool()
    tool2 = DummyEchoTool()

    registry.register(tool1)

    # Re-registering without overwrite=True should raise ValueError
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool2, overwrite=False)

    # Re-registering with overwrite=True succeeds
    registry.register(tool2, overwrite=True)
    assert registry.get("dummy_echo") is tool2


def test_registry_invalid_type_registration():
    registry = ToolRegistry()
    with pytest.raises(TypeError, match="Expected Tool instance"):
        registry.register("not_a_tool")  # type: ignore


def test_registry_unregister():
    registry = ToolRegistry()
    tool = DummyEchoTool()
    registry.register(tool)

    removed = registry.unregister("dummy_echo")
    assert removed is tool
    assert registry.has("dummy_echo") is False
    assert registry.unregister("dummy_echo") is None


def test_registry_get_schemas():
    registry = ToolRegistry()
    registry.register(DummyEchoTool())
    registry.register(FailingTool())

    schemas = registry.get_schemas()
    assert len(schemas) == 2
    names = [s["name"] for s in schemas]
    assert "dummy_echo" in names
    assert "failing_tool" in names


def test_application_tool_catalog_has_valid_descriptors():
    from core.tool_setup import create_tool_registry

    registry = create_tool_registry()
    assert registry.list_names()
    for tool in registry.list_tools():
        schema = tool.get_schema()
        assert isinstance(tool.name, str) and tool.name.strip()
        assert isinstance(tool.description, str) and tool.description.strip()
        assert tool.permission_level in set(PermissionLevel)
        assert schema["name"] == tool.name
        assert schema["description"] == tool.description
        assert isinstance(schema["parameters"], dict)
        assert schema["parameters"].get("type") == "object"
        assert isinstance(schema["parameters"].get("properties", {}), dict)


def test_registry_execute_existing_tool():
    registry = ToolRegistry()
    registry.register(DummyEchoTool())

    result = registry.execute("dummy_echo", text="testing registry dispatch")
    assert result.success is True
    assert result.data == {"echoed": "testing registry dispatch"}


def test_registry_execute_unknown_tool():
    registry = ToolRegistry()
    result = registry.execute("non_existent_tool", some_arg="value")
    assert result.success is False
    assert "not found" in result.error
    assert "not registered" in result.message


def test_permission_levels():
    assert PermissionLevel.SAFE.value == "SAFE"
    assert PermissionLevel.CONFIRM.value == "CONFIRM"
    assert PermissionLevel.DANGEROUS.value == "DANGEROUS"


def test_tool_result_sanitizes_secrets():
    res = ToolResult.ok(
        message="Created key: api_key=secret12345",
        data={"token": "AIzaSyD-fakeGoogleApiKey1234567890123"},
    )
    d = res.to_dict()
    assert "secret12345" not in d["message"]
    assert "REDACTED" in d["message"]
    assert "AIzaSyD" not in d["data"]["token"]
    assert "REDACTED" in d["data"]["token"]
