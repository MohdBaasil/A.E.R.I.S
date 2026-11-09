from unittest.mock import MagicMock, patch

from ai.provider import GeminiProvider
from config import Config
from main import run_voice_mode
from voice.speech_to_text import SpeechToText
from voice.text_to_speech import TextToSpeech


def test_stt_applies_conversational_end_of_phrase_settings():
    with patch("voice.speech_to_text.sr.Microphone"):
        stt = SpeechToText(timeout=3, phrase_time_limit=8, pause_threshold=0.4, non_speaking_duration=0.2)
    assert stt.timeout == 3
    assert stt.phrase_time_limit == 8
    assert stt.recognizer.pause_threshold == 0.4
    assert stt.recognizer.non_speaking_duration == 0.2


def test_config_loads_voice_performance_options(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("AI_API_KEY", "fake")
    monkeypatch.setenv("AI_MODEL", "fake")
    monkeypatch.setenv("VOICE_DEBUG", "true")
    monkeypatch.setenv("VOICE_MAX_HISTORY_MESSAGES", "12")
    monkeypatch.setenv("TTS_VOLUME", "0.6")
    config = Config.load()
    assert config.voice_debug is True
    assert config.voice_max_history_messages == 12
    assert config.tts_volume == 0.6


def test_provider_reuses_its_session_and_honors_retry_limit():
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
    session.post.return_value = response
    provider = GeminiProvider("key", "model", retries=0, session=session)
    assert provider.generate([{"role": "user", "content": "hello"}]) == "ok"
    assert provider._session is session
    assert session.post.call_count == 1


def test_provider_applies_explicit_connect_and_read_timeouts():
    provider = GeminiProvider("key", "model", connect_timeout=2, read_timeout=7)
    assert provider._timeout_tuple == (2, 7)


def test_tts_engine_is_reused_for_multiple_utterances():
    with patch("voice.text_to_speech.pyttsx3.init") as init:
        engine = MagicMock()
        init.return_value = engine
        tts = TextToSpeech()
        tts.speak("one")
        tts.speak("two")
    init.assert_called_once()
    assert engine.say.call_count == 2


def test_voice_debug_instruments_without_speaking_thinking(capsys):
    assistant = MagicMock()
    assistant.send.return_value = "A natural response."
    assistant.last_timings = {"llm_seconds": 0.01}
    stt = MagicMock()
    stt.device_name = "Test microphone"
    stt.device_index = 0
    stt.last_timings = {"stt_seconds": 0.01}
    stt.listen.side_effect = ["hello", "exit"]
    tts = MagicMock()
    tts.last_speak_seconds = 0.01
    run_voice_mode(assistant, stt, tts, voice_debug=True)
    spoken = [call.args[0] for call in tts.speak.call_args_list]
    assert "Thinking..." not in spoken
    assert "A natural response." in spoken
    output = capsys.readouterr().out
    assert "[VOICE DEBUG]" in output
    assert "STT: 0.01s" in output
    assert "LLM: 0.01s" in output
    assert "Tools: 0.00s" in output
    assert "TTS: 0.01s" in output
    assert "Total:" in output
