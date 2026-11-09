"""Natural-language buffering for incremental voice responses."""

from __future__ import annotations

import re


class SentenceBuffer:
    """Accumulate small model chunks until a natural speech boundary is ready."""

    def __init__(
        self,
        min_chars: int = 40,
        max_chars: int = 180,
        flush_timeout: float = 0.8,
    ) -> None:
        self.min_chars = max(1, min_chars)
        self.max_chars = max(self.min_chars, max_chars)
        self.flush_timeout = max(0.1, flush_timeout)
        self._buffer = ""

    def feed(self, chunk: str) -> list[str]:
        """Add one chunk and return complete natural-language speech segments."""
        if not chunk or not isinstance(chunk, str):
            return []
        self._buffer += chunk
        ready: list[str] = []
        while len(self._buffer) >= self.min_chars:
            boundary = self._find_boundary()
            if boundary is None and len(self._buffer) < self.max_chars:
                break
            cut = boundary if boundary is not None else self._word_boundary()
            segment = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:].lstrip()
            if segment:
                ready.append(segment)
        return ready

    def flush(self) -> list[str]:
        """Return any final partial sentence once the stream has ended."""
        final = self._buffer.strip()
        self._buffer = ""
        return [final] if final else []

    def clear(self) -> None:
        """Discard pending text, used when a tool protocol block starts or stream is interrupted."""
        self._buffer = ""

    def _find_boundary(self) -> int | None:
        matches = list(re.finditer(r"[.!?;:](?:\s+|$)", self._buffer))
        for match in matches:
            if match.end() >= self.min_chars:
                return match.end()
        return None

    def _word_boundary(self) -> int:
        limit = min(len(self._buffer), self.max_chars)
        cut = self._buffer.rfind(" ", 0, limit)
        return cut + 1 if cut > 0 else limit
