"""
Tests for AERIS Phase 3E — Tool Router, Confirmation, and Assistant Integration.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from core.confirmation import (
    AutoApproveHandler,
    AutoDenyHandler,
    ConfirmationHandler,
    TextConfirmationHandler,
    VoiceConfirmationHandler,
)
from core.tool_router import RouterResult, ToolCall, ToolRouter
from core.tool_setup import create_tool_registry
from core.assistant import Assistant
from tools.base import Tool, ToolResult
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyToolSafe(Tool):
    """A SAFE read-only tool for testing."""
    name = "test_safe_tool"
    description = "Returns a greeting."
    parameters = {
        "properties": {"name": {"type": "string", "description": "Name to greet"}},
        "required": ["name"],
    }
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):
        name = kwargs.get("name", "World")
        return ToolResult.ok(message=f"Hello, {name}!", data={"greeted": name})


class DummyToolConfirm(Tool):
    """A CONFIRM tool for testing permission gates."""
    name = "test_confirm_tool"
    description = "Does something that needs confirmation."
    parameters = {}
    permission_level = PermissionLevel.CONFIRM

    def execute(self, **kwargs):
        return ToolResult.ok(message="Confirmed action done.")


class DummyToolDangerous(Tool):
    """A DANGEROUS tool for testing permission gates."""
    name = "test_dangerous_tool"
    description = "Deletes something dangerous."
    parameters = {}
    permission_level = PermissionLevel.DANGEROUS

    def execute(self, **kwargs):
        return ToolResult.ok(message="Dangerous action done.")


class DummyToolFails(Tool):
    """A tool that always raises an exception."""
    name = "test_failing_tool"
    description = "Always crashes."
    parameters = {}
    permission_level = PermissionLevel.SAFE

    def execute(self, **kwargs):
        raise RuntimeError("Intentional test crash")


def _make_tool_call_block(tool: str, args: dict) -> str:
    """Create a model output containing a tool_call block."""
    payload = json.dumps({"tool": tool, "arguments": args})
    return f"Sure, let me do that.\n```tool_call\n{payload}\n```\nDone!"


# ---------------------------------------------------------------------------
# Tests: ToolCall and RouterResult dataclasses
# ---------------------------------------------------------------------------

class TestToolCallDataclass(unittest.TestCase):
    def test_defaults(self):
        call = ToolCall(tool="foo")
        self.assertEqual(call.tool, "foo")
        self.assertEqual(call.arguments, {})

    def test_with_arguments(self):
        call = ToolCall(tool="bar", arguments={"x": 1})
        self.assertEqual(call.arguments, {"x": 1})


class TestRouterResultDataclass(unittest.TestCase):
    def test_defaults(self):
        result = RouterResult()
        self.assertFalse(result.is_tool_call)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.tool_results, [])
        self.assertEqual(result.text_response, "")
        self.assertIsNone(result.error)


# ---------------------------------------------------------------------------
# Tests: Confirmation Handlers
# ---------------------------------------------------------------------------

class TestAutoApproveHandler(unittest.TestCase):
    def test_always_approves(self):
        handler = AutoApproveHandler()
        self.assertTrue(handler.request_confirmation("tool", "desc", "CONFIRM"))
        self.assertTrue(handler.request_confirmation("tool", "desc", "DANGEROUS"))


class TestAutoDenyHandler(unittest.TestCase):
    def test_always_denies(self):
        handler = AutoDenyHandler()
        self.assertFalse(handler.request_confirmation("tool", "desc", "CONFIRM"))
        self.assertFalse(handler.request_confirmation("tool", "desc", "DANGEROUS"))


class TestTextConfirmationHandler(unittest.TestCase):
    @patch("builtins.input", return_value="yes")
    def test_user_approves(self, mock_input):
        handler = TextConfirmationHandler()
        self.assertTrue(handler.request_confirmation("tool", "Do something", "CONFIRM"))

    @patch("builtins.input", return_value="no")
    def test_user_denies(self, mock_input):
        handler = TextConfirmationHandler()
        self.assertFalse(handler.request_confirmation("tool", "Do something", "CONFIRM"))

    @patch("builtins.input", side_effect=EOFError)
    def test_eof_denies(self, mock_input):
        handler = TextConfirmationHandler()
        self.assertFalse(handler.request_confirmation("tool", "Do something", "CONFIRM"))


class TestVoiceConfirmationHandler(unittest.TestCase):
    def test_voice_approves(self):
        mock_tts = MagicMock()
        mock_stt = MagicMock()
        mock_stt.listen.return_value = "yes"
        handler = VoiceConfirmationHandler(tts=mock_tts, stt=mock_stt)
        self.assertTrue(handler.request_confirmation("tool", "Open something", "CONFIRM"))
        mock_tts.speak.assert_called_once()

    def test_voice_denies(self):
        mock_tts = MagicMock()
        mock_stt = MagicMock()
        mock_stt.listen.return_value = "no"
        handler = VoiceConfirmationHandler(tts=mock_tts, stt=mock_stt)
        self.assertFalse(handler.request_confirmation("tool", "Open something", "CONFIRM"))


# ---------------------------------------------------------------------------
# Tests: ToolRouter — Parsing
# ---------------------------------------------------------------------------

class TestToolRouterParsing(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.router = ToolRouter(self.registry)

    def test_no_tool_calls(self):
        calls = self.router.parse_tool_calls("Just a normal response.")
        self.assertEqual(calls, [])

    def test_single_tool_call(self):
        model_output = _make_tool_call_block("open_application", {"app_name": "notepad"})
        calls = self.router.parse_tool_calls(model_output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "open_application")
        self.assertEqual(calls[0].arguments, {"app_name": "notepad"})

    def test_multiple_tool_calls(self):
        block1 = '```tool_call\n{"tool": "tool_a", "arguments": {"x": 1}}\n```'
        block2 = '```tool_call\n{"tool": "tool_b", "arguments": {"y": 2}}\n```'
        model_output = f"Step 1\n{block1}\nStep 2\n{block2}"
        calls = self.router.parse_tool_calls(model_output)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].tool, "tool_a")
        self.assertEqual(calls[1].tool, "tool_b")

    def test_malformed_json_skipped(self):
        model_output = "```tool_call\n{invalid json}\n```"
        calls = self.router.parse_tool_calls(model_output)
        self.assertEqual(calls, [])

    def test_missing_tool_field_skipped(self):
        model_output = '```tool_call\n{"arguments": {"x": 1}}\n```'
        calls = self.router.parse_tool_calls(model_output)
        self.assertEqual(calls, [])

    def test_strip_tool_blocks(self):
        model_output = 'Before.\n```tool_call\n{"tool": "x", "arguments": {}}\n```\nAfter.'
        stripped = self.router.strip_tool_blocks(model_output)
        self.assertIn("Before.", stripped)
        self.assertIn("After.", stripped)
        self.assertNotIn("tool_call", stripped)


# ---------------------------------------------------------------------------
# Tests: ToolRouter — Execution
# ---------------------------------------------------------------------------

class TestToolRouterExecution(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(DummyToolSafe())
        self.registry.register(DummyToolConfirm())
        self.registry.register(DummyToolDangerous())
        self.registry.register(DummyToolFails())

    def test_execute_safe_tool(self):
        router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())
        call = ToolCall(tool="test_safe_tool", arguments={"name": "AERIS"})
        result = router.execute_tool_call(call)
        self.assertTrue(result.success)
        self.assertIn("AERIS", result.message)

    def test_registered_system_info_is_safe_and_executes_without_confirmation(self):
        registry = create_tool_registry()
        system_info = registry.get("system_info")
        system_info.execute = MagicMock(return_value=ToolResult.ok("CPU metrics collected."))
        confirmation = MagicMock()
        router = ToolRouter(registry, confirmation_handler=confirmation)

        result = router.execute_tool_call(ToolCall(tool="system_info"))

        self.assertEqual(system_info.permission_level, PermissionLevel.SAFE)
        self.assertTrue(result.success)
        system_info.execute.assert_called_once_with()
        confirmation.request_confirmation.assert_not_called()

    def test_confirm_and_dangerous_tools_still_request_confirmation(self):
        confirmation = MagicMock()
        confirmation.request_confirmation.return_value = True
        router = ToolRouter(self.registry, confirmation_handler=confirmation)

        for tool_name, permission in (
            ("test_confirm_tool", PermissionLevel.CONFIRM),
            ("test_dangerous_tool", PermissionLevel.DANGEROUS),
        ):
            with self.subTest(tool=tool_name):
                result = router.execute_tool_call(ToolCall(tool=tool_name))
                self.assertTrue(result.success)
                self.assertEqual(confirmation.request_confirmation.call_args.kwargs["permission_level"], permission.value)

        self.assertEqual(confirmation.request_confirmation.call_count, 2)

    def test_execute_unknown_tool_fails(self):
        router = ToolRouter(self.registry)
        call = ToolCall(tool="nonexistent_tool", arguments={})
        result = router.execute_tool_call(call)
        self.assertFalse(result.success)
        self.assertIn("not available", result.message)

    def test_confirm_tool_approved(self):
        router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())
        call = ToolCall(tool="test_confirm_tool", arguments={})
        result = router.execute_tool_call(call)
        self.assertTrue(result.success)

    def test_confirm_tool_denied(self):
        router = ToolRouter(self.registry, confirmation_handler=AutoDenyHandler())
        call = ToolCall(tool="test_confirm_tool", arguments={})
        result = router.execute_tool_call(call)
        self.assertFalse(result.success)
        self.assertIn("declined", result.message)

    def test_dangerous_tool_approved(self):
        router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())
        call = ToolCall(tool="test_dangerous_tool", arguments={})
        result = router.execute_tool_call(call)
        self.assertTrue(result.success)

    def test_dangerous_tool_denied(self):
        router = ToolRouter(self.registry, confirmation_handler=AutoDenyHandler())
        call = ToolCall(tool="test_dangerous_tool", arguments={})
        result = router.execute_tool_call(call)
        self.assertFalse(result.success)

    def test_confirm_tool_no_handler_denies(self):
        router = ToolRouter(self.registry, confirmation_handler=None)
        call = ToolCall(tool="test_confirm_tool", arguments={})
        result = router.execute_tool_call(call)
        self.assertFalse(result.success)
        self.assertIn("confirmation is not available", result.message)

    def test_failing_tool_returns_error(self):
        router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())
        call = ToolCall(tool="test_failing_tool", arguments={})
        result = router.execute_tool_call(call)
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# Tests: ToolRouter — Full Pipeline (process_model_response)
# ---------------------------------------------------------------------------

class TestToolRouterPipeline(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(DummyToolSafe())
        self.router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())

    def test_conversational_response(self):
        result = self.router.process_model_response("Hello, how can I help you?")
        self.assertFalse(result.is_tool_call)
        self.assertIn("Hello", result.text_response)

    def test_tool_call_response(self):
        model_output = _make_tool_call_block("test_safe_tool", {"name": "World"})
        result = self.router.process_model_response(model_output)
        self.assertTrue(result.is_tool_call)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(len(result.tool_results), 1)
        self.assertTrue(result.tool_results[0].success)

    def test_tool_call_preserves_surrounding_text(self):
        model_output = _make_tool_call_block("test_safe_tool", {"name": "Test"})
        result = self.router.process_model_response(model_output)
        self.assertIn("Sure", result.text_response)
        self.assertNotIn("tool_call", result.text_response)


# ---------------------------------------------------------------------------
# Tests: tool_setup.py
# ---------------------------------------------------------------------------

class TestToolSetup(unittest.TestCase):
    @patch("core.tool_setup.register_file_tools")
    @patch("core.tool_setup.register_system_tools")
    @patch("core.tool_setup.register_application_tools")
    def test_creates_populated_registry(self, mock_app, mock_sys, mock_file):
        registry = create_tool_registry()
        self.assertIsInstance(registry, ToolRegistry)
        mock_app.assert_called_once()
        mock_sys.assert_called_once()
        mock_file.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: ToolRouter — Schema Text Generation
# ---------------------------------------------------------------------------

class TestToolRouterSchemaText(unittest.TestCase):
    def test_empty_registry(self):
        registry = ToolRegistry()
        router = ToolRouter(registry)
        text = router.get_tool_definitions_text()
        self.assertIn("No tools", text)

    def test_populated_registry(self):
        registry = ToolRegistry()
        registry.register(DummyToolSafe())
        router = ToolRouter(registry)
        text = router.get_tool_definitions_text()
        self.assertIn("test_safe_tool", text)
        self.assertIn("SAFE", text)
        self.assertIn("name", text)  # parameter listed


# ---------------------------------------------------------------------------
# Tests: Assistant Integration with ToolRouter
# ---------------------------------------------------------------------------

class TestAssistantToolIntegration(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(DummyToolSafe())

    def test_assistant_without_tool_router(self):
        """Phase 1 backward compatibility: no ToolRouter, pure passthrough."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "I'm just a chatbot."
        assistant = Assistant(mock_provider, tool_router=None)
        reply = assistant.send("Hello")
        self.assertEqual(reply, "I'm just a chatbot.")

    def test_assistant_conversational_with_tool_router(self):
        """Conversational response passes through unmodified."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Sure, I can help with that."
        router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())
        assistant = Assistant(mock_provider, tool_router=router)
        reply = assistant.send("What is the weather?")
        self.assertIn("Sure", reply)

    def test_assistant_tool_call_flow(self):
        """Tool call → execution → result fed back → final response."""
        mock_provider = MagicMock()
        tool_block = _make_tool_call_block("test_safe_tool", {"name": "User"})

        # First call: model returns tool_call. Second call: model returns summary.
        mock_provider.generate.side_effect = [
            tool_block,
            "I greeted User successfully.",
        ]
        router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())
        assistant = Assistant(mock_provider, tool_router=router)
        reply = assistant.send("Greet User")
        self.assertIn("greeted", reply)
        self.assertEqual(mock_provider.generate.call_count, 2)

    def test_assistant_handles_provider_error(self):
        """Provider error is caught gracefully."""
        from ai.provider import ProviderError
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = ProviderError("Network error")
        router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())
        assistant = Assistant(mock_provider, tool_router=router)
        reply = assistant.send("Hello")
        self.assertIn("trouble reaching the AI service", reply)

    def test_assistant_system_prompt_includes_tools(self):
        """When a ToolRouter is provided, the system prompt includes tool definitions."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "OK"
        router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())
        assistant = Assistant(mock_provider, tool_router=router)
        # Check the initial history contains tool info
        system_content = assistant._history[0]["content"]
        self.assertIn("test_safe_tool", system_content)
        self.assertIn("tool_call", system_content)

    def test_assistant_reset(self):
        """Reset clears conversation but keeps system prompt."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Hi"
        router = ToolRouter(self.registry, confirmation_handler=AutoApproveHandler())
        assistant = Assistant(mock_provider, tool_router=router)
        assistant.send("Hello")
        self.assertGreater(len(assistant._history), 2)
        assistant.reset()
        self.assertEqual(len(assistant._history), 2)


if __name__ == "__main__":
    unittest.main()
