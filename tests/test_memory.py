from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from ai.provider import AIProvider
from core.assistant import Assistant
from core.memory import (
    MemoryCategory,
    MemoryLimitError,
    MemoryStore,
    SensitiveMemoryError,
    parse_explicit_memory_command,
)
from core.tool_router import ToolRouter
from tools.base import Tool, ToolResult
from tools.applications import OpenApplicationTool
from tools.permissions import PermissionLevel
from tools.registry import ToolRegistry


class CaptureProvider(AIProvider):
    def __init__(self, replies=None):
        self.replies = list(replies or ["ok"])
        self.requests = []

    def generate(self, messages):
        self.requests.append(list(messages))
        return self.replies.pop(0) if self.replies else "ok"


def test_memory_save_retrieve_reopen_update_and_delete(tmp_path):
    database = tmp_path / "memory.sqlite3"
    store = MemoryStore(database)
    saved = store.save("My preferred editor is VS Code.", MemoryCategory.PREFERENCE, memory_key="preference:editor")
    assert saved.id and saved.created_at == saved.updated_at
    assert store.retrieve("Open my editor")[0].id == saved.id

    reopened = MemoryStore(database)
    persisted = reopened.retrieve("preferred editor")
    assert len(persisted) == 1
    assert persisted[0].content == "My preferred editor is VS Code."

    updated = reopened.save("My preferred editor is PyCharm.", MemoryCategory.PREFERENCE, memory_key="preference:editor")
    assert updated.id == saved.id
    assert updated.created_at == saved.created_at
    assert updated.updated_at >= saved.updated_at
    assert len(reopened.list_memories()) == 1
    assert reopened.delete(updated.id)
    assert reopened.retrieve("preferred editor") == []


def test_memory_retrieval_is_relevant_multi_match_and_bounded(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path / "memory.db")
    store.save("My preferred editor is VS Code.", "preference", memory_key="preference:editor")
    store.save("The CollabSpace project uses Python.", "project", memory_key="project:collabspace")
    store.save("I prefer a dark theme.", "preference", memory_key="preference:theme")

    assert len(store.retrieve("What is my preferred editor?")) == 1
    assert len(store.retrieve("Tell me about my preferences", limit=2)) == 2
    assert store.retrieve("Preferred editor", limit=1000)[0].content == "My preferred editor is VS Code."
    assert store.retrieve("completely unrelated topic") == []
    import core.memory as memory_module
    monkeypatch.setattr(memory_module, "MAX_RETRIEVAL_RESULTS", 1)
    assert len(store.retrieve("Tell me about my preferences", limit=50)) == 1


def test_memory_store_enforces_growth_and_content_bounds(tmp_path, monkeypatch):
    import core.memory as memory_module

    monkeypatch.setattr(memory_module, "MAX_MEMORIES", 2)
    store = MemoryStore(tmp_path / "memory.db")
    store.save("fact one", "fact")
    store.save("fact two", "fact")
    with pytest.raises(MemoryLimitError):
        store.save("fact three", "fact")
    with pytest.raises(ValueError):
        store.save("x" * 1001, "fact")


@pytest.mark.parametrize("sensitive", [
    "My API key is secret-value-123456",
    "My password is correct-horse-battery-staple",
    "My access token is ghp_abcdefghijklmnopqrstuvwxyz123456",
    "-----BEGIN RSA PRIVATE KEY-----\nsecret text",
    "My credit card number is 4111 1111 1111 1111",
])
def test_memory_rejects_sensitive_content_without_logging_or_persisting(tmp_path, caplog, sensitive):
    store = MemoryStore(tmp_path / "memory.db")
    caplog.set_level(logging.INFO, logger="aeris.memory")
    with pytest.raises(SensitiveMemoryError):
        store.save(sensitive, "fact")
    assert store.list_memories() == []
    assert sensitive not in caplog.text


def test_memory_rejects_secrets_in_metadata(tmp_path, caplog):
    store = MemoryStore(tmp_path / "memory.db")
    secret_metadata = {"provider": "test", "credential": "sk-test-memory-secret-1234567890"}
    caplog.set_level(logging.INFO, logger="aeris.memory")
    with pytest.raises(SensitiveMemoryError):
        store.save("A harmless note.", "fact", metadata=secret_metadata)
    assert store.count() == 0
    assert "sk-test-memory-secret" not in caplog.text


def test_explicit_remember_recall_forget_and_update_are_deterministic(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    provider = CaptureProvider()
    assistant = Assistant(provider, memory_store=store)

    assert "remember that" in assistant.send("Remember that my preferred editor is VS Code.").lower()
    assert assistant.send("What is my preferred editor?") == "You told me: My preferred editor is VS Code."
    assert "PyCharm" in assistant.send("Actually, I prefer PyCharm.")
    assert len(store.list_memories()) == 1
    assert store.list_memories()[0].content == "My preferred editor is PyCharm."
    assert assistant.send("Forget that I use PyCharm.") == "I forgot the matching saved memory."
    assert assistant.send("What is my preferred editor?") == "I don't have a saved memory about that."
    assert provider.requests == []


@pytest.mark.parametrize("phrase", ["Delete file", "Delete disposable-test.txt", "Delete my project folder"])
def test_file_deletion_phrases_are_not_treated_as_memory_commands(phrase):
    assert parse_explicit_memory_command(phrase) is None


def test_remember_simple_fact_and_recall_all_and_forget_project(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    assistant = Assistant(CaptureProvider(), memory_store=store)
    assistant.send("Remember that I use VS Code.")
    assistant.send("Remember that the CollabSpace project uses Python.")
    answer = assistant.send("What do you remember about me?")
    assert "VS Code" in answer and "CollabSpace" in answer
    assert assistant.send("Forget everything about the CollabSpace project.") == "I forgot the matching saved memory."
    assert len(store.list_memories()) == 1


def test_explicit_secret_request_is_redacted_from_history_provider_and_logs(tmp_path, caplog):
    secret = "Remember that my API key is sk-very-secret-987654321"
    store = MemoryStore(tmp_path / "memory.db")
    provider = CaptureProvider()
    assistant = Assistant(provider, memory_store=store)
    caplog.set_level(logging.INFO, logger="aeris.memory")

    response = assistant.send(secret)

    assert "can't save" in response
    assert secret not in repr(assistant.history)
    assert "very-secret" not in caplog.text
    assert store.list_memories() == []
    assert provider.requests == []


def test_only_relevant_memory_is_added_to_ephemeral_provider_context(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.save("My preferred editor is VS Code.", "preference", memory_key="preference:editor")
    store.save("The CollabSpace project uses Python.", "project", memory_key="project:collabspace")
    provider = CaptureProvider(["An answer."])
    assistant = Assistant(provider, memory_store=store)

    assert assistant.send("Tell me what editor I prefer.") == "An answer."
    sent = provider.requests[0]
    memory_messages = [message["content"] for message in sent if "RELEVANT SAVED USER MEMORY" in message["content"]]
    assert len(memory_messages) == 1
    assert "VS Code" in memory_messages[0]
    assert "CollabSpace" not in memory_messages[0]
    assert not any("RELEVANT SAVED USER MEMORY" in item["content"] for item in assistant.history)


@patch("tools.applications.subprocess.Popen")
def test_saved_editor_preference_resolves_open_my_editor_safely(mock_popen, tmp_path):
    mock_popen.return_value.pid = 9876
    store = MemoryStore(tmp_path / "memory.db")
    store.save("I use VS Code.", "preference", memory_key="preference:editor")
    registry = ToolRegistry()
    registry.register(OpenApplicationTool())
    provider = CaptureProvider(["Opened your editor."])
    assistant = Assistant(provider, ToolRouter(registry), memory_store=store)

    response = assistant.send("Open my editor.")

    assert response == "Opened Visual Studio Code."
    mock_popen.assert_called_once()
    assert len(provider.requests) == 0


def test_memory_context_is_not_persisted_or_mixed_with_tool_results(tmp_path):
    class ReadTool(Tool):
        name = "safe_value"
        description = "Returns a test value."
        parameters = {"type": "object", "properties": {}}
        permission_level = PermissionLevel.SAFE

        def execute(self, **kwargs):
            return ToolResult.ok("fresh tool data", data={"value": "42"})

    store = MemoryStore(tmp_path / "memory.db")
    store.save("My preferred editor is VS Code.", "preference", memory_key="preference:editor")
    registry = ToolRegistry()
    registry.register(ReadTool())
    provider = CaptureProvider([
        '```tool_call\n{"tool":"safe_value","arguments":{}}\n```',
        "The value is 42.",
    ])
    assistant = Assistant(provider, ToolRouter(registry), memory_store=store)
    assistant.send("Tell me my editor and call safe_value.")

    tool_result = next(item["content"] for item in assistant.history if "```tool_result" in item["content"])
    assert '"value": "42"' in tool_result
    assert "VS Code" not in tool_result
    assert len(store.list_memories()) == 1
    assert any("RELEVANT SAVED USER MEMORY" in item["content"] for item in provider.requests[1])


def test_diagnostics_show_metadata_without_memory_contents(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    record = store.save("My preferred editor is VS Code.", "preference", memory_key="preference:editor")
    assistant = Assistant(CaptureProvider(), memory_store=store)

    output = assistant.send("Memory diagnostics")

    assert "1 saved memories" in output
    assert str(store.path) in output
    assert "storage limit: 500" in output
    assert "retrieval limit: 10" in output
    assert record.content not in output
    assert record.id not in output


def test_explicit_memory_list_shows_saved_contents(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.save("My preferred editor is VS Code.", "preference", memory_key="preference:editor")
    assistant = Assistant(CaptureProvider(), memory_store=store)

    assert "VS Code" in assistant.send("Show my saved memories.")


@pytest.mark.parametrize(("phrase", "expected"), [
    ("Remember that my editor is VS Code.", ("save", "my editor is VS Code")),
    ("What do you remember about my editor?", ("recall", "my editor")),
    ("Show my saved memories.", ("list_memories", "")),
    ("Forget my editor preference.", ("forget", "my editor preference")),
    ("Forget everything you remember about me.", ("clear", "")),
    ("Update my editor preference to Cursor.", ("update_editor", "Cursor")),
    ("AERIS memory status", ("diagnostics", "")),
    ("Export my memories.", ("export", "")),
])
def test_memory_management_command_phrases(phrase, expected):
    assert parse_explicit_memory_command(phrase) == expected


def test_update_editor_and_forget_all_are_deterministic(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    assistant = Assistant(CaptureProvider(), memory_store=store)
    assistant.send("Remember that my preferred editor is VS Code.")
    assert "Cursor" in assistant.send("Update my editor preference to Cursor.")
    assert assistant.send("What do you remember about my editor?") == "You told me: My preferred editor is Cursor."
    assert assistant.send("Forget all memories.") == "I forgot all saved memories (1 removed)."
    assert store.count() == 0


def test_export_json_is_local_valid_and_never_overwrites(tmp_path):
    import json

    store = MemoryStore(tmp_path / "memory.sqlite3")
    record = store.save("My preferred editor is VS Code.", "preference", memory_key="preference:editor")
    assistant = Assistant(CaptureProvider(), memory_store=store)
    response = assistant.send("Export my memories")
    export_path = tmp_path / "aeris_memory_export.json"
    assert str(export_path) in response
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["format_version"] == 1
    assert exported["memories"][0]["content"] == record.content

    repeated_export = assistant.send("Export my memories")
    assert "already exists" in repeated_export
    assert json.loads(export_path.read_text(encoding="utf-8"))["memories"][0]["content"] == record.content


def test_listing_and_export_include_all_records_up_to_configured_capacity(tmp_path):
    import json

    store = MemoryStore(tmp_path / "memory.db", max_records=60)
    for index in range(60):
        store.save(f"Saved fact number {index}.", "fact", memory_key=f"fact:{index}")
    assert len(store.list_memories()) == 60
    exported_path = store.export_json()
    exported = json.loads(exported_path.read_text(encoding="utf-8"))
    assert len(exported["memories"]) == 60


def test_configured_retention_refuses_new_records_without_deleting(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", max_records=1)
    store.save("I like tea.", "fact")
    with pytest.raises(MemoryLimitError, match="Nothing was deleted"):
        store.save("I like coffee.", "fact")
    assert store.count() == 1
    assert store.list_memories()[0].content == "I like tea."


def test_forget_specific_editor_memory_keeps_unrelated_preference(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.save("My preferred editor is VS Code.", "preference", memory_key="preference:editor")
    store.save("I prefer a dark theme.", "preference", memory_key="preference:theme")
    assistant = Assistant(CaptureProvider(), memory_store=store)
    assert assistant.send("Forget my editor preference.") == "I forgot the matching saved memory."
    assert store.count() == 1
    assert store.retrieve("theme")


def test_ordinary_conversation_does_not_create_memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    assistant = Assistant(CaptureProvider(["I'll keep that in mind."]), memory_store=store)
    assistant.send("I usually use VS Code.")
    assert store.count() == 0


def test_reset_and_history_pruning_do_not_delete_persistent_memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.save("My preferred editor is VS Code.", "preference", memory_key="preference:editor")
    assistant = Assistant(CaptureProvider(), memory_store=store, max_history_messages=4)
    for i in range(8):
        assistant.send(f"ordinary message {i}")
    assistant.reset()
    assert len(assistant.history) == 2
    assert store.retrieve("preferred editor")


def test_explicit_memory_commands_work_in_streaming_mode(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    assistant = Assistant(CaptureProvider(), memory_store=store)
    assert "remember" in "".join(assistant.send_stream("Remember that my preferred editor is VS Code.")).lower()
    assert "VS Code" in "".join(assistant.send_stream("What is my preferred editor?"))
