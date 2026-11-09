"""Convert an assistant reply into safe, natural speech without changing text output."""

from __future__ import annotations

import json
import re

_TOOL_BLOCK = re.compile(r"```(?:tool_call|tool_result)\s*.*?```", re.IGNORECASE | re.DOTALL)
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def prepare_speech_response(text: str) -> str:
    """Remove protocol and formatting that should never be read aloud."""
    cleaned = _TOOL_BLOCK.sub("", text or "")
    had_code_block = bool(_CODE_BLOCK.search(cleaned))
    cleaned = _CODE_BLOCK.sub("", cleaned)
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE.sub(lambda match: "the code" if "(" in match.group(1) else match.group(1), cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if had_code_block and cleaned:
        cleaned = f"{cleaned} I've prepared the code."
    elif had_code_block:
        cleaned = "I've prepared the code."

    # A raw JSON response is internal/protocol-like and never useful spoken aloud.
    if cleaned.startswith(("{", "[")):
        try:
            json.loads(cleaned)
            return "I completed the requested action."
        except (TypeError, ValueError):
            pass
    return cleaned or "I’m sorry, I wasn’t able to prepare a spoken response."
