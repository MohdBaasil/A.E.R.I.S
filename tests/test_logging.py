"""
Tests for AERIS logging infrastructure — Phase 10F.

Covers:
- setup_logging() creates a RotatingFileHandler with correct maxBytes / backupCount.
- Calling setup_logging() twice does NOT duplicate handlers.
- Log level is applied correctly.
- Config dataclass exposes log_max_bytes, log_backup_count, log_file with correct defaults.
- Config.load() parses LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_FILE env vars.
- Exception tracebacks are captured by logger.exception().
"""

from __future__ import annotations

import logging
import logging.handlers
from logging.handlers import RotatingFileHandler
import os

import pytest

from main import setup_logging
from config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_root_handlers():
    """
    Remove any RotatingFileHandler / FileHandler from the root logger
    before and after each test to prevent handler pollution across tests.
    """
    root = logging.getLogger()
    # Capture original handlers so we can restore state
    original = list(root.handlers)

    # Pre-clean
    for h in list(root.handlers):
        if isinstance(h, (logging.FileHandler, RotatingFileHandler)):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    yield

    # Post-clean
    for h in list(root.handlers):
        if isinstance(h, (logging.FileHandler, RotatingFileHandler)):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# setup_logging() — handler creation
# ---------------------------------------------------------------------------

def test_setup_logging_creates_rotating_handler(tmp_path):
    """setup_logging() must add exactly one RotatingFileHandler."""
    log_file = str(tmp_path / "test.log")
    setup_logging(log_file=log_file, max_bytes=1024, backup_count=3)

    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1, "Expected exactly one RotatingFileHandler"


def test_setup_logging_handler_max_bytes(tmp_path):
    """RotatingFileHandler.maxBytes must match the max_bytes argument."""
    log_file = str(tmp_path / "test.log")
    setup_logging(log_file=log_file, max_bytes=65536, backup_count=2)

    root = logging.getLogger()
    handler = next(h for h in root.handlers if isinstance(h, RotatingFileHandler))
    assert handler.maxBytes == 65536


def test_setup_logging_handler_backup_count(tmp_path):
    """RotatingFileHandler.backupCount must match the backup_count argument."""
    log_file = str(tmp_path / "test.log")
    setup_logging(log_file=log_file, max_bytes=1024, backup_count=7)

    root = logging.getLogger()
    handler = next(h for h in root.handlers if isinstance(h, RotatingFileHandler))
    assert handler.backupCount == 7


def test_setup_logging_handler_is_rotating_not_plain(tmp_path):
    """The handler must be RotatingFileHandler, not a plain FileHandler."""
    log_file = str(tmp_path / "test.log")
    setup_logging(log_file=log_file)

    root = logging.getLogger()
    plain_handlers = [
        h for h in root.handlers
        if type(h) is logging.FileHandler  # exact type, not subclass
    ]
    assert plain_handlers == [], "Expected no plain FileHandler (only RotatingFileHandler)"


def test_setup_logging_creates_log_file(tmp_path):
    """setup_logging() must create the log file on disk."""
    log_file = str(tmp_path / "test.log")
    setup_logging(log_file=log_file)

    root = logging.getLogger()
    # Flush to ensure file is created
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler):
            h.emit(logging.LogRecord("test", logging.INFO, "", 0, "ping", (), None))
            h.flush()

    assert os.path.exists(log_file), f"Log file was not created at {log_file}"


# ---------------------------------------------------------------------------
# setup_logging() — duplicate-handler prevention
# ---------------------------------------------------------------------------

def test_setup_logging_no_duplicate_on_second_call(tmp_path):
    """Calling setup_logging() twice must not duplicate file handlers."""
    log_file = str(tmp_path / "test.log")
    setup_logging(log_file=log_file)
    setup_logging(log_file=log_file)

    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1, (
        f"Expected 1 RotatingFileHandler after two calls, got {len(file_handlers)}"
    )


def test_setup_logging_no_duplicate_on_many_calls(tmp_path):
    """Calling setup_logging() many times must keep exactly one handler."""
    log_file = str(tmp_path / "test.log")
    for _ in range(5):
        setup_logging(log_file=log_file)

    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1


# ---------------------------------------------------------------------------
# setup_logging() — log level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level_str, expected", [
    ("DEBUG",   logging.DEBUG),
    ("INFO",    logging.INFO),
    ("WARNING", logging.WARNING),
    ("ERROR",   logging.ERROR),
])
def test_setup_logging_applies_log_level(tmp_path, level_str, expected):
    """setup_logging() must set the root logger to the requested level."""
    log_file = str(tmp_path / "test.log")
    setup_logging(log_level=level_str, log_file=log_file)

    root = logging.getLogger()
    assert root.level == expected, (
        f"Root logger level should be {expected} for '{level_str}', got {root.level}"
    )


def test_setup_logging_handler_inherits_level(tmp_path):
    """The RotatingFileHandler's level must match the requested log level."""
    log_file = str(tmp_path / "test.log")
    setup_logging(log_level="WARNING", log_file=log_file)

    root = logging.getLogger()
    handler = next(h for h in root.handlers if isinstance(h, RotatingFileHandler))
    assert handler.level == logging.WARNING


# ---------------------------------------------------------------------------
# Config — new log rotation fields
# ---------------------------------------------------------------------------

def test_config_default_log_max_bytes():
    """Config must expose log_max_bytes defaulting to 2 MiB."""
    config = Config.load()
    assert config.log_max_bytes == 2 * 1024 * 1024


def test_config_default_log_backup_count():
    """Config must expose log_backup_count defaulting to 5."""
    config = Config.load()
    assert config.log_backup_count == 5


def test_config_default_log_file():
    """Config must expose log_file defaulting to 'aeris.log'."""
    config = Config.load()
    assert config.log_file == "aeris.log"


def test_config_log_max_bytes_from_env(monkeypatch):
    """LOG_MAX_BYTES env var must override log_max_bytes."""
    monkeypatch.setenv("LOG_MAX_BYTES", "4194304")
    config = Config.load()
    assert config.log_max_bytes == 4_194_304


def test_config_log_backup_count_from_env(monkeypatch):
    """LOG_BACKUP_COUNT env var must override log_backup_count."""
    monkeypatch.setenv("LOG_BACKUP_COUNT", "10")
    config = Config.load()
    assert config.log_backup_count == 10


def test_config_log_file_from_env(monkeypatch):
    """LOG_FILE env var must override log_file."""
    monkeypatch.setenv("LOG_FILE", "/tmp/custom_aeris.log")
    config = Config.load()
    assert config.log_file == "/tmp/custom_aeris.log"


# ---------------------------------------------------------------------------
# Exception logging
# ---------------------------------------------------------------------------

def test_logger_exception_captures_traceback(tmp_path, caplog):
    """logger.exception() must include exception information in the log record."""
    log_file = str(tmp_path / "test.log")
    setup_logging(log_level="DEBUG", log_file=log_file)

    test_logger = logging.getLogger("aeris.test_exception")
    with caplog.at_level(logging.ERROR, logger="aeris.test_exception"):
        try:
            raise ValueError("test error for AERIS logging test")
        except ValueError:
            test_logger.exception("Caught expected test error: %s", "detail")

    assert any("Caught expected test error" in r.message for r in caplog.records), (
        "exception message not found in log records"
    )
    assert any(r.exc_info is not None for r in caplog.records), (
        "exc_info (traceback) was not attached to the log record"
    )

