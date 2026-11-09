"""Local wake-word and active-conversation state for voice mode."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass


def normalize_voice_text(text: str) -> str:
    """Normalize recognised speech for local command matching."""
    return re.sub(r"\s+", " ", text.strip().lower().strip(".,!?;:"))


@dataclass
class WakeWordController:
    """Small, replaceable activation policy built on recognised local text."""

    enabled: bool = False
    wake_word: str = "aeris"
    active_window_seconds: float = 10.0
    _active_until: float = 0.0

    def __post_init__(self) -> None:
        self.wake_word = normalize_voice_text(self.wake_word) or "aeris"
        self.active_window_seconds = max(1.0, float(self.active_window_seconds))

    @property
    def is_active(self) -> bool:
        return not self.enabled or time.monotonic() < self._active_until

    def reset(self) -> None:
        self._active_until = 0.0

    def refresh(self) -> None:
        if self.enabled:
            self._active_until = time.monotonic() + self.active_window_seconds

    def consume(self, text: str) -> tuple[bool, str, bool]:
        """Return ``(accepted, request, wake_only)`` for recognised speech."""
        cleaned = normalize_voice_text(text)
        if not self.enabled:
            return bool(cleaned), text.strip(), False

        if self.is_active:
            self.refresh()
            return bool(cleaned), text.strip(), False

        pattern = rf"(?:^|\b)(?:hey\s+)?{re.escape(self.wake_word)}(?:\b|$)"
        match = re.search(pattern, cleaned)
        if not match:
            return False, "", False

        request = (cleaned[:match.start()] + " " + cleaned[match.end():]).strip(" ,.!?;:")
        self.refresh()
        return True, request, not request
