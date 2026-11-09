import pytest
from unittest.mock import patch, MagicMock, call

from main import is_exit_command, run_text_mode, run_voice_mode, main


def test_is_exit_command():
    assert is_exit_command("exit") is True
    assert is_exit_command("QUIT") is True
    assert is_exit_command("Goodbye") is True
    assert is_exit_command("bye!") is True
    assert is_exit_command("goodbye aeris") is True
    assert is_exit_command("quit now please") is True
    assert is_exit_command("shutdown") is True
    assert is_exit_command("SHUTDOWN") is True
    assert is_exit_command("shutdown aeris") is True
    assert is_exit_command("stop aeris") is True
    assert is_exit_command("STOP AERIS") is True
    assert is_exit_command("stop aeris please") is True
    assert is_exit_command("shut down aeris") is True
    assert is_exit_command("stop") is True
    assert is_exit_command("hello aeris") is False
    assert is_exit_command("what is the weather today?") is False
    assert is_exit_command("") is False


def test_run_text_mode_conversation_and_exit():
    mock_assistant = MagicMock()
    mock_assistant.send.return_value = "I am doing well."

    # Simulate user inputs: "hello", then "exit"
    with patch("builtins.input", side_effect=["hello", "exit"]):
        run_text_mode(mock_assistant)

    mock_assistant.send.assert_called_once_with("hello")


def test_run_voice_mode_conversation_and_exit():
    mock_assistant = MagicMock()
    mock_assistant.send.return_value = "AERIS is ready."

    mock_stt = MagicMock()
    mock_stt.device_name = "Microphone Array"
    mock_stt.device_index = 1
    # Simulate: silence first, then recognized speech, then voice exit command
    mock_stt.listen.side_effect = ["", "hello aeris", "goodbye"]

    mock_tts = MagicMock()

    run_voice_mode(mock_assistant, mock_stt, mock_tts)

    # Verify greeting + assistant response + exit confirmation
    mock_assistant.send.assert_called_once_with("hello aeris")
    assert mock_tts.speak.call_count == 3
    mock_tts.speak.assert_has_calls([
        call("AERIS online. Voice mode active."),
        call("AERIS is ready."),
        call("Goodbye."),
    ])
    mock_stt.close.assert_called_once()
    mock_tts.stop.assert_called_once()


def test_run_voice_mode_handles_exceptions_gracefully():
    mock_assistant = MagicMock()
    mock_stt = MagicMock()
    mock_stt.device_name = "Microphone Array"
    mock_stt.device_index = 1
    mock_stt.listen.side_effect = [Exception("PyAudio glitch"), "quit"]

    mock_tts = MagicMock()

    # Should handle the exception without crashing and exit on 'quit'
    run_voice_mode(mock_assistant, mock_stt, mock_tts)

    assert mock_stt.listen.call_count == 2
    mock_stt.close.assert_called_once()
    mock_tts.stop.assert_called_once()


@patch("main.run_text_mode")
def test_run_voice_mode_mic_unavailable_falls_back_to_text(mock_run_text):
    mock_assistant = MagicMock()
    mock_stt = MagicMock()
    mock_stt.microphone = None
    mock_tts = MagicMock()

    run_voice_mode(mock_assistant, mock_stt, mock_tts)

    mock_run_text.assert_called_once_with(mock_assistant)


@patch("main.run_text_mode")
def test_run_voice_mode_consecutive_errors_falls_back_to_text(mock_run_text):
    mock_assistant = MagicMock()
    mock_stt = MagicMock()
    mock_stt.device_name = "Microphone Array"
    mock_stt.device_index = 1
    mock_stt.listen.side_effect = [
        Exception("Error 1"),
        Exception("Error 2"),
        Exception("Error 3"),
        Exception("Error 4"),
        Exception("Error 5"),
    ]
    mock_tts = MagicMock()

    run_voice_mode(mock_assistant, mock_stt, mock_tts)

    mock_run_text.assert_called_once_with(mock_assistant)


@patch("main.setup_logging")
@patch("main.Config.load")
@patch("main.run_provider_test", return_value=0)
def test_main_provider_test_flag(mock_provider_test, mock_load_config, mock_logging):
    mock_load_config.return_value = MagicMock()
    with patch("sys.argv", ["main.py", "--provider-test"]):
        exit_code = main()
    assert exit_code == 0
    mock_provider_test.assert_called_once()


@patch("main.setup_logging")
@patch("main.Config.load")
@patch("main.create_provider")
@patch("main.SpeechToText")
@patch("main.TextToSpeech")
@patch("main.run_voice_mode")
def test_main_voice_mode_flag(mock_run_voice, mock_tts_cls, mock_stt_cls, mock_create_provider, mock_load_config, mock_logging):
    mock_config = MagicMock()
    mock_load_config.return_value = mock_config
    mock_provider = MagicMock()
    mock_create_provider.return_value = mock_provider

    mock_stt = MagicMock()
    mock_stt.microphone = MagicMock()
    mock_stt_cls.return_value = mock_stt

    mock_tts = MagicMock()
    mock_tts_cls.return_value = mock_tts

    with patch("sys.argv", ["main.py", "--voice"]):
        exit_code = main()

    assert exit_code == 0
    mock_run_voice.assert_called_once()


@patch("main.setup_logging")
@patch("main.Config.load")
@patch("main.create_provider")
@patch("main.SpeechToText")
@patch("main.TextToSpeech")
@patch("main.run_text_mode")
def test_main_voice_mode_mic_failure_fallback_to_text(mock_run_text, mock_tts_cls, mock_stt_cls, mock_create_provider, mock_load_config, mock_logging):
    mock_config = MagicMock()
    mock_load_config.return_value = mock_config
    mock_provider = MagicMock()
    mock_create_provider.return_value = mock_provider

    mock_stt = MagicMock()
    mock_stt.microphone = None
    mock_stt_cls.return_value = mock_stt

    mock_tts = MagicMock()
    mock_tts_cls.return_value = mock_tts

    with patch("sys.argv", ["main.py", "--voice"]):
        exit_code = main()

    assert exit_code == 0
    mock_run_text.assert_called_once()


@patch("main.setup_logging")
@patch("main.Config.load")
@patch("main.create_provider")
@patch("main.SpeechToText", side_effect=Exception("PyAudio driver failed"))
@patch("main.TextToSpeech")
@patch("main.run_text_mode")
def test_main_voice_mode_init_exception_fallback_to_text(mock_run_text, mock_tts_cls, mock_stt_cls, mock_create_provider, mock_load_config, mock_logging):
    mock_config = MagicMock()
    mock_load_config.return_value = mock_config
    mock_provider = MagicMock()
    mock_create_provider.return_value = mock_provider

    mock_tts = MagicMock()
    mock_tts_cls.return_value = mock_tts

    with patch("sys.argv", ["main.py", "--voice"]):
        exit_code = main()

    assert exit_code == 0
    mock_run_text.assert_called_once()

