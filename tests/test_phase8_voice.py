from unittest.mock import MagicMock, patch

from main import is_tts_interrupt_command, run_voice_mode
from voice.activation import WakeWordController
from voice.response_cleanup import prepare_speech_response
from voice.text_to_speech import TextToSpeech


def test_wake_word_disabled_accepts_normal_speech():
    controller = WakeWordController(enabled=False)
    assert controller.consume("open vscode") == (True, "open vscode", False)


def test_wake_word_ignores_unrelated_speech_when_idle():
    controller = WakeWordController(enabled=True, wake_word="aeris")
    assert controller.consume("hello everyone") == (False, "", False)


def test_wake_word_accepts_inline_request_and_keeps_active_window():
    controller = WakeWordController(enabled=True, wake_word="aeris", active_window_seconds=10)
    assert controller.consume("Hey AERIS, open VS Code") == (True, "open vs code", False)
    accepted, request, wake_only = controller.consume("open my project")
    assert (accepted, request, wake_only) == (True, "open my project", False)


def test_wake_word_only_activation_requests_follow_up():
    controller = WakeWordController(enabled=True)
    assert controller.consume("AERIS") == (True, "", True)


def test_expired_active_window_returns_to_idle():
    controller = WakeWordController(enabled=True)
    controller.refresh()
    controller._active_until = 0
    assert controller.consume("open vscode") == (False, "", False)


def test_voice_loop_wake_word_ignores_idle_speech_and_keeps_context():
    assistant = MagicMock()
    assistant.send.return_value = "Opening VS Code."
    stt = MagicMock()
    stt.device_name, stt.device_index = "Test mic", 0
    stt.listen.side_effect = ["background conversation", "aeris", "open VS Code", "exit"]
    tts = MagicMock()

    run_voice_mode(assistant, stt, tts, wake_word_enabled=True)

    assistant.send.assert_called_once_with("open VS Code")
    spoken = [call.args[0] for call in tts.speak.call_args_list]
    assert "Yes?" in spoken


def test_interrupt_phrases_are_local():
    for phrase in ("stop", "stop talking", "quiet", "be quiet", "cancel", "enough"):
        assert is_tts_interrupt_command(phrase)


def test_speech_response_cleanup_removes_protocol_and_markdown():
    raw = "**Done**\n```tool_result\n{\"secret\": \"no\"}\n```\n- Open [Docs](https://example.com)"
    assert prepare_speech_response(raw) == "Done Open Docs"


def test_speech_response_cleanup_replaces_raw_json():
    assert prepare_speech_response('{"tool": "open_application"}') == "I completed the requested action."


def test_unavailable_tts_voice_falls_back_to_default():
    with patch("voice.text_to_speech.pyttsx3.init") as init:
        engine = MagicMock()
        engine.getProperty.return_value = [MagicMock(id="installed", name="Installed")]
        init.return_value = engine
        TextToSpeech(voice="missing voice")
    voice_calls = [call for call in engine.setProperty.call_args_list if call.args[0] == "voice"]
    assert voice_calls == []


def test_tts_async_reuses_engine_and_stop_is_safe():
    with patch("voice.text_to_speech.pyttsx3.init") as init:
        engine = MagicMock()
        init.return_value = engine
        tts = TextToSpeech()
        assert tts.speak_async("Hello") is True
        tts.stop()
    init.assert_called_once()
    engine.stop.assert_called()
