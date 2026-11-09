"""
Phase 6 tests — True Voice-to-Voice Assistant.

Covers:
- is_voice_to_text_command() classification
- Local command interception (exit / voice-to-text) never calls Gemini
- Full voice conversation loop: STT → Assistant → TTS
- Voice loop continues after successful response
- Empty / unintelligible speech does not call Assistant
- "I didn't catch that" spoken after MAX_SILENCE_TTS consecutive silences
- Gemini timeout does not crash voice mode
- Gemini connection failure does not crash voice mode
- TTS failure does not crash voice mode (session continues, text printed)
- Microphone error during session falls back to text mode
- Voice mode cleans up STT and TTS resources even on exception
- _speak() helper prints and speaks (or just prints when TTS is None/fails)
- Existing ToolRouter behaviour remains intact through voice interface
- provider timeout tuple is enforced (unit)
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, call

import requests

from main import (
    is_exit_command,
    is_voice_to_text_command,
    run_voice_mode,
    run_text_mode,
    _speak,
    VOICE_TO_TEXT_PHRASES,
)
from ai.provider import GeminiProvider, ProviderError


# ===========================================================================
# is_voice_to_text_command()
# ===========================================================================

class TestIsVoiceToTextCommand:
    def test_deactivate_voice(self):
        assert is_voice_to_text_command("deactivate voice") is True

    def test_disable_voice(self):
        assert is_voice_to_text_command("disable voice") is True

    def test_stop_voice_mode(self):
        assert is_voice_to_text_command("stop voice mode") is True

    def test_switch_to_text(self):
        assert is_voice_to_text_command("switch to text") is True

    def test_switch_to_text_mode(self):
        assert is_voice_to_text_command("switch to text mode") is True

    def test_text_mode(self):
        assert is_voice_to_text_command("text mode") is True

    def test_turn_off_voice(self):
        assert is_voice_to_text_command("turn off voice") is True

    def test_stop_listening(self):
        assert is_voice_to_text_command("stop listening") is True

    def test_case_insensitive(self):
        assert is_voice_to_text_command("DEACTIVATE VOICE") is True
        assert is_voice_to_text_command("Text Mode") is True

    def test_trailing_punctuation(self):
        assert is_voice_to_text_command("deactivate voice.") is True

    def test_non_voice_command(self):
        assert is_voice_to_text_command("hello aeris") is False
        assert is_voice_to_text_command("what time is it") is False
        assert is_voice_to_text_command("exit") is False
        assert is_voice_to_text_command("") is False

    def test_all_documented_phrases_recognised(self):
        """Every phrase in VOICE_TO_TEXT_PHRASES must be recognised."""
        for phrase in VOICE_TO_TEXT_PHRASES:
            assert is_voice_to_text_command(phrase) is True, f"'{phrase}' not recognised"


# ===========================================================================
# _speak() helper
# ===========================================================================

class TestSpeakHelper:
    def test_speaks_and_prints_with_tts(self, capsys):
        logger = MagicMock()
        mock_tts = MagicMock()
        _speak(mock_tts, "Hello there", logger)
        mock_tts.speak.assert_called_once_with("Hello there")
        captured = capsys.readouterr()
        assert "Hello there" in captured.out

    def test_prints_even_without_tts(self, capsys):
        logger = MagicMock()
        _speak(None, "No TTS response", logger)
        captured = capsys.readouterr()
        assert "No TTS response" in captured.out

    def test_tts_failure_does_not_raise(self, capsys):
        logger = MagicMock()
        mock_tts = MagicMock()
        mock_tts.speak.side_effect = RuntimeError("TTS engine crashed")
        # Must not raise
        _speak(mock_tts, "Still here", logger)
        logger.error.assert_called_once()
        captured = capsys.readouterr()
        assert "Still here" in captured.out


# ===========================================================================
# Voice conversation loop — core behaviours
# ===========================================================================

class TestVoiceConversationLoop:
    """Full voice loop: STT → Assistant → TTS, then back to listening."""

    def _make_stt(self):
        stt = MagicMock()
        stt.device_name = "Microphone Array"
        stt.device_index = 0
        return stt

    def test_full_conversation_and_exit(self):
        """Voice loop: greeting spoken, speech recognised, reply spoken, exit clean."""
        mock_assistant = MagicMock()
        mock_assistant.send.return_value = "I am AERIS. How can I help?"

        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = ["Hello AERIS", "exit"]

        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        # Greeting + reply + goodbye = 3 TTS calls
        assert mock_tts.speak.call_count == 3
        mock_tts.speak.assert_has_calls([
            call("AERIS online. Voice mode active."),
            call("I am AERIS. How can I help?"),
            call("Goodbye."),
        ])
        mock_assistant.send.assert_called_once_with("Hello AERIS")
        mock_stt.close.assert_called_once()
        mock_tts.stop.assert_called_once()

    def test_loop_continues_after_response(self):
        """After a successful reply the loop listens again (no premature exit)."""
        mock_assistant = MagicMock()
        mock_assistant.send.return_value = "Sure thing."

        mock_stt = self._make_stt()
        # Two real messages then exit
        mock_stt.listen.side_effect = ["question one", "question two", "quit"]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        assert mock_assistant.send.call_count == 2
        mock_assistant.send.assert_any_call("question one")
        mock_assistant.send.assert_any_call("question two")

    def test_empty_speech_does_not_call_assistant(self):
        """Empty STT result must not reach assistant.send()."""
        mock_assistant = MagicMock()
        mock_stt = self._make_stt()
        # Two silences then exit
        mock_stt.listen.side_effect = ["", "", "exit"]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        mock_assistant.send.assert_not_called()

    def test_unrecognised_speech_does_not_crash(self):
        """Whitespace-only input is treated as silence — no crash, no Gemini call."""
        mock_assistant = MagicMock()
        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = ["   ", "bye"]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        mock_assistant.send.assert_not_called()

    def test_repeated_silence_triggers_tts_after_threshold(self):
        """After MAX_SILENCE_TTS consecutive empty results 'I didn't catch that' is spoken."""
        mock_assistant = MagicMock()
        mock_stt = self._make_stt()
        # 3 silences (triggers the message at the 3rd) then exit
        mock_stt.listen.side_effect = ["", "", "", "exit"]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        speak_texts = [c.args[0] for c in mock_tts.speak.call_args_list]
        assert "I didn't catch that." in speak_texts
        mock_assistant.send.assert_not_called()


# ===========================================================================
# Local command interception — NEVER calls Gemini
# ===========================================================================

class TestLocalCommandInterception:
    """Exit and voice-to-text commands must be handled locally without Gemini."""

    def _make_stt(self):
        stt = MagicMock()
        stt.device_name = "Microphone Array"
        stt.device_index = 0
        return stt

    @pytest.mark.parametrize("phrase", ["exit", "quit", "goodbye", "shutdown", "stop aeris"])
    def test_exit_commands_never_call_gemini(self, phrase):
        mock_assistant = MagicMock()
        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = [phrase]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        mock_assistant.send.assert_not_called()

    @pytest.mark.parametrize("phrase", [
        "deactivate voice",
        "disable voice",
        "switch to text",
        "switch to text mode",
        "stop voice mode",
        "text mode",
        "turn off voice",
        "stop listening",
    ])
    @patch("main.run_text_mode")
    def test_voice_to_text_commands_never_call_gemini(self, mock_run_text, phrase):
        mock_assistant = MagicMock()
        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = [phrase]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        mock_assistant.send.assert_not_called()
        mock_run_text.assert_called_once_with(mock_assistant)

    @patch("main.run_text_mode")
    def test_voice_to_text_switches_confirmation_handler(self, mock_run_text):
        """When switching to text mode the confirmation handler updates."""
        mock_tool_router = MagicMock()
        mock_assistant = MagicMock()
        mock_assistant._tool_router = mock_tool_router

        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = ["deactivate voice"]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        mock_tool_router.set_confirmation_handler.assert_called_once()
        mock_run_text.assert_called_once_with(mock_assistant)


# ===========================================================================
# AI / Network failure resilience
# ===========================================================================

class TestVoiceAIFailureResilience:
    """Gemini errors must not crash the voice session."""

    def _make_stt(self):
        stt = MagicMock()
        stt.device_name = "Microphone Array"
        stt.device_index = 0
        return stt

    def test_gemini_timeout_voice_continues(self):
        """ProviderError (timeout) → AERIS speaks error msg, loop continues."""
        mock_assistant = MagicMock()
        # First call raises ProviderError (as assistant.send does internally)
        mock_assistant.send.side_effect = [
            "I'm having trouble reaching the AI service. Please try again.",
            None,  # unreachable — exit handled next
        ]
        mock_stt = self._make_stt()
        # One real input that causes the (already-wrapped) error, then exit
        mock_stt.listen.side_effect = ["what time is it", "exit"]
        mock_tts = MagicMock()

        # Must not raise
        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        # TTS called for greeting + error message + goodbye
        assert mock_tts.speak.call_count >= 2
        speak_texts = [c.args[0] for c in mock_tts.speak.call_args_list]
        assert any("timed out" in t or "couldn't reach" in t or "trouble" in t for t in speak_texts)

    def test_gemini_connection_error_voice_continues(self):
        """assistant.send() returning an error string → session survives."""
        mock_assistant = MagicMock()
        mock_assistant.send.return_value = (
            "I'm having trouble reaching the AI service. Please try again."
        )
        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = ["hello", "exit"]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        assert mock_tts.speak.call_count >= 2  # greeting + reply
        mock_stt.close.assert_called_once()

    def test_unexpected_assistant_exception_voice_continues(self):
        """If assistant.send() raises unexpectedly, loop recovers and continues."""
        mock_assistant = MagicMock()
        mock_assistant.send.side_effect = RuntimeError("Completely unexpected crash")

        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = ["cause the crash", "exit"]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        mock_stt.close.assert_called_once()
        mock_tts.stop.assert_called_once()


# ===========================================================================
# TTS failure resilience
# ===========================================================================

class TestVoiceTTSFailureResilience:
    """TTS errors must not crash the voice session."""

    def _make_stt(self):
        stt = MagicMock()
        stt.device_name = "Microphone Array"
        stt.device_index = 0
        return stt

    def test_tts_failure_session_continues(self, capsys):
        """If TTS.speak() raises, the response is still printed and loop goes on."""
        mock_assistant = MagicMock()
        mock_assistant.send.return_value = "Here is the answer."

        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = ["ask something", "exit"]

        mock_tts = MagicMock()
        # Greeting OK, then reply fails, then goodbye OK
        mock_tts.speak.side_effect = [None, RuntimeError("Speaker error"), None]

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        captured = capsys.readouterr()
        assert "Here is the answer." in captured.out
        mock_stt.close.assert_called_once()


# ===========================================================================
# Microphone failure / fallback
# ===========================================================================

class TestVoiceMicrophoneFailure:
    @patch("main.run_text_mode")
    def test_mic_unavailable_at_start_falls_back(self, mock_run_text):
        mock_assistant = MagicMock()
        mock_stt = MagicMock()
        mock_stt.microphone = None
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)
        mock_run_text.assert_called_once_with(mock_assistant)

    @patch("main.run_text_mode")
    def test_consecutive_mic_errors_fall_back_to_text(self, mock_run_text):
        mock_assistant = MagicMock()
        mock_stt = MagicMock()
        mock_stt.device_name = "Microphone Array"
        mock_stt.device_index = 0
        mock_stt.listen.side_effect = [
            Exception("PyAudio crash 1"),
            Exception("PyAudio crash 2"),
            Exception("PyAudio crash 3"),
            Exception("PyAudio crash 4"),
            Exception("PyAudio crash 5"),
        ]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)
        mock_run_text.assert_called_once_with(mock_assistant)


# ===========================================================================
# Resource cleanup
# ===========================================================================

class TestVoiceResourceCleanup:
    def _make_stt(self):
        stt = MagicMock()
        stt.device_name = "Microphone Array"
        stt.device_index = 0
        return stt

    def test_resources_closed_on_normal_exit(self):
        mock_assistant = MagicMock()
        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = ["exit"]
        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        mock_stt.close.assert_called_once()
        mock_tts.stop.assert_called_once()

    def test_resources_closed_after_consecutive_errors(self):
        mock_assistant = MagicMock()
        mock_stt = self._make_stt()
        mock_stt.listen.side_effect = [Exception("err")] * 5
        mock_tts = MagicMock()

        with patch("main.run_text_mode"):
            run_voice_mode(mock_assistant, mock_stt, mock_tts)

        mock_stt.close.assert_called_once()
        mock_tts.stop.assert_called_once()


# ===========================================================================
# GeminiProvider timeout enforcement
# ===========================================================================

class TestGeminiProviderTimeout:
    """Unit-level: verify the timeout tuple is built and passed correctly."""

    def test_timeout_tuple_uses_explicit_connect_and_read(self):
        provider = GeminiProvider(api_key="fake", model="gemini-test", timeout_seconds=20)
        connect, read = provider._timeout_tuple
        assert read == 20
        assert connect >= 5  # at least the minimum
        assert connect < read  # connect is a fraction of read

    def test_default_timeout_tuple(self):
        provider = GeminiProvider(api_key="fake", model="gemini-test")
        connect, read = provider._timeout_tuple
        assert read == 30
        assert connect == max(5, int(30 * GeminiProvider._CONNECT_FRACTION))

    @patch("ai.provider.requests.Session.post")
    def test_requests_receives_tuple_timeout(self, mock_post):
        """requests.post must be called with a tuple, never a plain int."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}}]
        }
        mock_post.return_value = mock_response

        provider = GeminiProvider(api_key="fake", model="gemini-test", timeout_seconds=15)
        provider.generate([{"role": "user", "content": "hi"}])

        _, kwargs = mock_post.call_args
        timeout_arg = kwargs.get("timeout")
        assert isinstance(timeout_arg, tuple), f"Expected tuple, got {type(timeout_arg)}"
        assert len(timeout_arg) == 2

    @patch("ai.provider.requests.Session.post", side_effect=requests.exceptions.Timeout("Read timed out"))
    def test_timeout_raises_provider_error(self, mock_post):
        """requests.Timeout must surface as ProviderError after retries."""
        provider = GeminiProvider(api_key="fake", model="gemini-test", timeout_seconds=1)
        with pytest.raises(ProviderError, match="timed out"):
            provider.generate([{"role": "user", "content": "hi"}])

    @patch("ai.provider.requests.Session.post", side_effect=requests.exceptions.ConnectionError("refused"))
    def test_connection_error_raises_provider_error(self, mock_post):
        """requests.ConnectionError must surface as ProviderError after retries."""
        provider = GeminiProvider(api_key="fake", model="gemini-test", timeout_seconds=1)
        with pytest.raises(ProviderError, match="connect"):
            provider.generate([{"role": "user", "content": "hi"}])


# ===========================================================================
# ToolRouter still works through voice interface
# ===========================================================================

class TestToolRouterIntactThroughVoice:
    """Smoke-test that the tool pipeline is not bypassed in voice mode."""

    def test_voice_mode_uses_same_assistant_pipeline(self):
        """run_voice_mode must call assistant.send() — it does not bypass the pipeline."""
        mock_assistant = MagicMock()
        mock_assistant.send.return_value = "Tool executed successfully."

        mock_stt = MagicMock()
        mock_stt.device_name = "Microphone Array"
        mock_stt.device_index = 0
        mock_stt.listen.side_effect = ["open notepad", "exit"]

        mock_tts = MagicMock()

        run_voice_mode(mock_assistant, mock_stt, mock_tts)

        # assistant.send() (which internally calls ToolRouter) was called
        mock_assistant.send.assert_called_once_with("open notepad")
