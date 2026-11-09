"""
Confirmation abstraction for AERIS tool execution.

Handles user confirmation for CONFIRM and DANGEROUS permission levels.
Works in both text mode and voice mode.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("aeris.core.confirmation")


class ConfirmationHandler(ABC):
    """Abstract base for requesting user confirmation before tool execution."""

    @abstractmethod
    def request_confirmation(self, tool_name: str, description: str, permission_level: str) -> bool:
        """
        Ask the user whether to proceed with a tool action.

        Args:
            tool_name: The tool being invoked.
            description: Human-readable description of what the tool will do.
            permission_level: "CONFIRM" or "DANGEROUS".

        Returns:
            True if the user approves, False otherwise.
        """
        raise NotImplementedError


class TextConfirmationHandler(ConfirmationHandler):
    """Confirmation via terminal input()."""

    def request_confirmation(self, tool_name: str, description: str, permission_level: str) -> bool:
        label = "[DANGEROUS]" if permission_level == "DANGEROUS" else "[CONFIRM]"
        print(f"\nAERIS {label}: {description}")
        try:
            answer = input("  Approve? (yes/no): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("yes", "y")


class VoiceConfirmationHandler(ConfirmationHandler):
    """Confirmation via TTS prompt and STT response."""

    def __init__(self, tts, stt) -> None:
        self._tts = tts
        self._stt = stt

    def request_confirmation(self, tool_name: str, description: str, permission_level: str) -> bool:
        label = "dangerous action" if permission_level == "DANGEROUS" else "action"
        prompt = f"I need your permission for a {label}: {description}. Do you approve? Say yes or no."
        print(f"\nAERIS [{permission_level}]: {description}")
        try:
            self._tts.speak(prompt)
            print("  Waiting for voice confirmation...")
            response = self._stt.listen().strip().lower()
            print(f"  You said: '{response}'")
            return response in ("yes", "yeah", "yep", "approve", "do it", "go ahead", "sure", "ok", "okay")
        except Exception as exc:
            logger.error("Voice confirmation failed: %s", exc)
            return False


class AutoApproveHandler(ConfirmationHandler):
    """Always approves. Used ONLY for testing."""

    def request_confirmation(self, tool_name: str, description: str, permission_level: str) -> bool:
        return True


class AutoDenyHandler(ConfirmationHandler):
    """Always denies. Used ONLY for testing."""

    def request_confirmation(self, tool_name: str, description: str, permission_level: str) -> bool:
        return False
