"""Bounded, local persistent user memory for AERIS."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("aeris.memory")

MAX_MEMORIES = 500
MAX_CONTENT_CHARS = 1000
MAX_RETRIEVAL_RESULTS = 10
MAX_DIAGNOSTIC_RESULTS = MAX_MEMORIES


class MemoryCategory(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    PROJECT = "project"
    WORKFLOW = "workflow"
    INSTRUCTION = "instruction"


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    content: str
    category: MemoryCategory
    created_at: str
    updated_at: str
    metadata: dict[str, Any]
    memory_key: Optional[str]


class MemoryError(ValueError):
    """Base error for memory validation and storage operations."""


class SensitiveMemoryError(MemoryError):
    """Raised when requested memory contains credentials or secrets."""


class MemoryLimitError(MemoryError):
    """Raised when the bounded memory store has reached capacity."""


_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)\b(api[_ -]?key|password|passcode|token|credential|private[_ -]?key|secret)\b"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\b(?:sk|ghp|gho|github_pat|xox[baprs])-?[A-Za-z0-9_-]{16,}\b", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:bank account|routing number|credit card|debit card|cvv|cvc)\b"),
    re.compile(r"(?i)\b(?:social security|ssn|passport number|medical record|health diagnosis)\b"),
)
_WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "about", "am", "be", "do", "does", "for", "from",
    "have", "i", "in", "is", "it", "me", "my", "of", "on", "that", "the", "this",
    "to", "use", "what", "you", "your", "remember", "forget", "everything", "everything",
}


def contains_sensitive_information(content: str) -> bool:
    return any(pattern.search(content) for pattern in _SENSITIVE_PATTERNS)


def _tokens(text: str) -> set[str]:
    tokens = {word.lower() for word in _WORD_PATTERN.findall(text)}
    canonical = {"preferred": "prefer", "preference": "prefer", "preferences": "prefer", "prefers": "prefer", "favourite": "favorite"}
    normalized = {canonical.get(word, word) for word in tokens}
    return {word for word in normalized if word not in _STOP_WORDS and len(word) > 1}


def classify_memory(content: str) -> MemoryCategory:
    text = content.casefold()
    if re.search(r"\b(prefer|preferred|preference|favorite|favourite|editor|ide|browser)\b", text) or re.search(
        r"\b(vs\s*code|vscode|pycharm|intellij|sublime|notepad\+\+)\b", text
    ):
        return MemoryCategory.PREFERENCE
    if re.search(r"\b(project|workspace|repo|repository)\b", text):
        return MemoryCategory.PROJECT
    if re.search(r"\b(workflow|routine|process|steps)\b", text):
        return MemoryCategory.WORKFLOW
    if re.search(r"\b(always|please|instruction|should|must)\b", text):
        return MemoryCategory.INSTRUCTION
    return MemoryCategory.FACT


def memory_key_for(content: str, category: MemoryCategory) -> Optional[str]:
    text = content.casefold()
    if category == MemoryCategory.PREFERENCE:
        if re.search(r"\b(editor|ide|vs\s*code|vscode|pycharm|intellij|sublime|notepad\+\+)\b", text):
            return "preference:editor"
        for topic in ("browser", "language", "theme", "timezone"):
            if re.search(rf"\b{topic}\b", text):
                return f"preference:{topic}"
    if category == MemoryCategory.PROJECT:
        match = re.search(r"\b(?:project|workspace|repository|repo)\s+([\w.-]+)", content, re.IGNORECASE)
        if match:
            return f"project:{match.group(1).casefold()}"
    normalized = " ".join(sorted(_tokens(content)))
    return f"{category.value}:{normalized}" if normalized else None


def parse_explicit_memory_command(text: str) -> Optional[tuple[str, str]]:
    """Recognize explicit memory commands; no model inference is involved."""
    if re.fullmatch(r"\s*(?:show|list)(?:\s+my|\s+the)?\s+(?:saved\s+)?memories\s*[?.!]*\s*", text, re.IGNORECASE) or re.fullmatch(
        r"\s*aeris\s+memory\s+list\s*[?.!]*\s*", text, re.IGNORECASE
    ):
        return "list_memories", ""
    if re.fullmatch(r"\s*(?:aeris\s+)?memory\s+(?:status|diagnostics)\s*[?.!]*\s*", text, re.IGNORECASE):
        return "diagnostics", ""
    if re.fullmatch(r"\s*(?:export|back\s*up)\s+(?:my\s+|the\s+)?(?:saved\s+)?memories\s*[?.!]*\s*", text, re.IGNORECASE):
        return "export", ""
    recall_all = re.fullmatch(
        r"\s*(?:what do you remember(?: about me)?|what do you know about me)\s*[?.!]*\s*",
        text,
        re.IGNORECASE,
    )
    if recall_all:
        return "recall_all", ""
    recall_topic = re.fullmatch(r"\s*what is my (?:preferred )?editor\s*[?.!]*\s*", text, re.IGNORECASE)
    if recall_topic:
        return "recall", "editor preference"
    recall_about = re.fullmatch(r"\s*what do you remember about (.+?)\s*[?.!]*\s*", text, re.IGNORECASE)
    if recall_about:
        return "recall", recall_about.group(1).strip()
    if re.fullmatch(r"\s*forget\s+(?:all|everything)(?:\s+(?:saved\s+)?memories)?\s*[?.!]*\s*", text, re.IGNORECASE) or re.fullmatch(
        r"\s*forget\s+everything\s+you\s+remember\s+about\s+me\s*[?.!]*\s*", text, re.IGNORECASE
    ):
        return "clear", ""
    save = re.fullmatch(r"\s*(?:please\s+)?(?:remember|save|store)(?:\s+that)?\s+(.+?)\s*[.!?]*\s*", text, re.IGNORECASE)
    if save:
        return "save", save.group(1).strip()
    editor_update = re.fullmatch(
        r"\s*update\s+my\s+(?:preferred\s+)?(?:editor|ide)(?:\s+preference)?\s+to\s+(.+?)\s*[.!?]*\s*",
        text,
        re.IGNORECASE,
    )
    if editor_update:
        return "update_editor", editor_update.group(1).strip()
    forget_everything = re.fullmatch(
        r"\s*(?:please\s+)?forget\s+(?:everything|all)(?: you remember)?\s+about\s+(.+?)\s*[.!?]*\s*",
        text,
        re.IGNORECASE,
    )
    if forget_everything:
        target = forget_everything.group(1).strip()
        return ("clear", "") if target.casefold() in {"me", "myself", "all"} else ("forget", target)
    delete_memory = re.fullmatch(
        r"\s*(?:please\s+)?delete\s+(?:the\s+)?memory(?:\s+about)?\s+(.+?)\s*[.!?]*\s*",
        text,
        re.IGNORECASE,
    )
    if delete_memory:
        return "forget", delete_memory.group(1).strip()
    forget = re.fullmatch(r"\s*(?:please\s+)?(?:forget|remove)(?:\s+that)?\s+(.+?)\s*[.!?]*\s*", text, re.IGNORECASE)
    if forget:
        return "forget", forget.group(1).strip()
    correction = re.fullmatch(
        r"\s*(?:actually,?\s+)?(?:i prefer|my preferred (?:editor|ide) is)\s+(.+?)\s*[.!?]*\s*",
        text,
        re.IGNORECASE,
    )
    if correction:
        value = correction.group(1).strip()
        if re.search(r"\b(vs\s*code|vscode|pycharm|intellij|sublime|notepad\+\+)\b", value, re.IGNORECASE):
            return "update_editor", value
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    """Small SQLite store with deterministic matching and bounded growth."""

    def __init__(self, path: str | Path, max_records: int | None = None) -> None:
        self.path = Path(path)
        self.max_records = max(1, int(MAX_MEMORIES if max_records is None else max_records))
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL CHECK(category IN ('preference','fact','project','workflow','instruction')),
                    memory_key TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
        logger.info("Local memory store initialized")

    def _connection(self):
        """Yield a fresh short-lived connection; DB access never leaks to providers/tools."""
        store = self

        class _ConnectionContext:
            def __enter__(self):
                store._lock.acquire()
                try:
                    self.connection = sqlite3.connect(store.path, timeout=5)
                    self.connection.row_factory = sqlite3.Row
                    return self.connection
                except Exception:
                    store._lock.release()
                    raise

            def __exit__(self, exc_type, exc, tb):
                try:
                    if exc_type is None:
                        self.connection.commit()
                    else:
                        self.connection.rollback()
                    self.connection.close()
                finally:
                    store._lock.release()

        return _ConnectionContext()

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            category=MemoryCategory(row["category"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"] or "{}"),
            memory_key=row["memory_key"],
        )

    def save(
        self,
        content: str,
        category: MemoryCategory | str = MemoryCategory.FACT,
        *,
        memory_key: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryRecord:
        content = content.strip() if isinstance(content, str) else ""
        if not content:
            raise MemoryError("Memory content must not be empty.")
        if len(content) > MAX_CONTENT_CHARS:
            raise MemoryError(f"Memory content cannot exceed {MAX_CONTENT_CHARS} characters.")
        if contains_sensitive_information(content):
            raise SensitiveMemoryError("I can’t save passwords, credentials, keys, tokens, or other secrets.")
        try:
            category = MemoryCategory(category)
        except ValueError as exc:
            raise MemoryError("Unsupported memory category.") from exc
        clean_metadata = metadata if isinstance(metadata, dict) else {}
        metadata_json = json.dumps(clean_metadata, sort_keys=True, separators=(",", ":"))
        if contains_sensitive_information(metadata_json):
            raise SensitiveMemoryError("I can't save passwords, credentials, keys, tokens, or other secrets.")
        timestamp = _now()
        record_id = str(uuid.uuid4())
        with self._connection() as connection:
            if memory_key:
                existing = connection.execute("SELECT * FROM memories WHERE memory_key = ?", (memory_key,)).fetchone()
                if existing:
                    connection.execute(
                        "UPDATE memories SET content=?, category=?, updated_at=?, metadata=? WHERE id=?",
                        (content, category.value, timestamp, metadata_json, existing["id"]),
                    )
                    row = connection.execute("SELECT * FROM memories WHERE id=?", (existing["id"],)).fetchone()
                    logger.info("User memory updated: id=%s category=%s", row["id"], category.value)
                    return self._record(row)
            count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if count >= self.max_records:
                raise MemoryLimitError(
                    f"Memory limit reached ({self.max_records} saved memories). Nothing was deleted; forget a memory before saving another."
                )
            connection.execute(
                "INSERT INTO memories(id, content, category, memory_key, created_at, updated_at, metadata) VALUES(?,?,?,?,?,?,?)",
                (record_id, content, category.value, memory_key, timestamp, timestamp, metadata_json),
            )
            row = connection.execute("SELECT * FROM memories WHERE id=?", (record_id,)).fetchone()
        logger.info("User memory saved: id=%s category=%s", record_id, category.value)
        return self._record(row)

    def update(self, memory_id: str, content: str, category: Optional[MemoryCategory | str] = None) -> Optional[MemoryRecord]:
        content = content.strip() if isinstance(content, str) else ""
        if not content or len(content) > MAX_CONTENT_CHARS:
            raise MemoryError("Memory content is empty or exceeds the size limit.")
        if contains_sensitive_information(content):
            raise SensitiveMemoryError("I can’t save passwords, credentials, keys, tokens, or other secrets.")
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                return None
            new_category = MemoryCategory(category).value if category is not None else row["category"]
            connection.execute(
                "UPDATE memories SET content=?, category=?, updated_at=? WHERE id=?",
                (content, new_category, _now(), memory_id),
            )
            updated = connection.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        logger.info("User memory updated: id=%s category=%s", memory_id, new_category)
        return self._record(updated)

    def delete(self, memory_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        if cursor.rowcount:
            logger.info("User memory deleted: id=%s", memory_id)
            return True
        return False

    def list_memories(self, limit: int = MAX_DIAGNOSTIC_RESULTS) -> list[MemoryRecord]:
        bounded = max(1, min(int(limit), MAX_DIAGNOSTIC_RESULTS))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC, id LIMIT ?", (bounded,)
            ).fetchall()
        return [
            self._record(row)
            for row in rows
            if not contains_sensitive_information(row["content"])
            and not contains_sensitive_information(row["metadata"] or "{}")
        ]

    def count(self) -> int:
        """Return the stored record count without exposing memory contents."""
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def export_json(self, destination: str | Path | None = None) -> Path:
        """Export saved records to a new local JSON file, never overwriting an existing file."""
        target = Path(destination) if destination is not None else self.path.parent / "aeris_memory_export.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        records = self.list_memories(limit=MAX_DIAGNOSTIC_RESULTS)
        document = {
            "format_version": 1,
            "exported_at": _now(),
            "memories": [
                {
                    "id": item.id,
                    "content": item.content,
                    "category": item.category.value,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                    "metadata": item.metadata,
                    "memory_key": item.memory_key,
                }
                for item in records
            ],
        }
        payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        descriptor = None
        created = False
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                descriptor = None
                stream.write(payload)
        except FileExistsError as exc:
            raise MemoryError("The export file already exists. Choose a different location or move the existing file first.") from exc
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            if created:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass
            raise MemoryError("The memory export could not be written.") from exc
        logger.info("Local memory export created: records=%d", len(records))
        return target

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        terms = _tokens(query)
        if not terms:
            return []
        bounded = max(1, min(int(limit), MAX_RETRIEVAL_RESULTS))
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM memories").fetchall()
        scored = []
        requested_topic = next((topic for topic in ("editor", "browser", "theme", "language", "timezone") if topic in terms), None)
        for row in rows:
            record = self._record(row)
            if contains_sensitive_information(record.content):
                continue
            if requested_topic and record.memory_key and record.memory_key.startswith("preference:"):
                if record.memory_key != f"preference:{requested_topic}":
                    continue
            overlap = len(terms & _tokens(record.content))
            if requested_topic and record.memory_key == f"preference:{requested_topic}":
                overlap += 2
            if overlap:
                scored.append((overlap, record.updated_at, record))
        scored.sort(key=lambda item: (-item[0], item[1], item[2].id))
        return [item[2] for item in scored[:bounded]]

    def forget_matching(self, query: str) -> int:
        terms = _tokens(query)
        if not terms:
            return 0
        with self._connection() as connection:
            rows = connection.execute("SELECT id, content FROM memories").fetchall()
            ids = [row["id"] for row in rows if terms <= _tokens(row["content"])]
            if ids:
                connection.executemany("DELETE FROM memories WHERE id=?", [(item,) for item in ids])
        if ids:
            logger.info("User memories deleted by explicit request: count=%d", len(ids))
        return len(ids)

    def clear(self) -> int:
        with self._connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            connection.execute("DELETE FROM memories")
        if count:
            logger.info("All user memories deleted: count=%d", count)
        return count
