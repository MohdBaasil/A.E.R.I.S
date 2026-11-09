import pytest
from unittest.mock import patch, MagicMock
import speech_recognition as sr
from voice.speech_to_text import SpeechToText

def test_stt_init_success():
    with patch('voice.speech_to_text.sr.Microphone') as mock_mic:
        stt = SpeechToText()
        mock_mic.assert_called_once()
        assert stt.microphone is not None
        assert stt.timeout == 4

def test_stt_init_mic_failure():
    with patch('voice.speech_to_text.sr.Microphone', side_effect=Exception("No Mic")):
        stt = SpeechToText()
        assert stt.microphone is None

def test_stt_listen_no_mic():
    with patch('voice.speech_to_text.sr.Microphone', side_effect=Exception("No Mic")):
        stt = SpeechToText()
        result = stt.listen()
        assert result == ""

def test_stt_listen_success():
    with patch('voice.speech_to_text.sr.Microphone') as mock_mic:
        stt = SpeechToText()
        
        # Mock the recognizer
        stt.recognizer = MagicMock()
        stt.recognizer.listen.return_value = "mock_audio"
        stt.recognizer.recognize_google.return_value = "hello aeris"
        
        result = stt.listen()
        assert result == "hello aeris"
        stt.recognizer.listen.assert_called_once()
        stt.recognizer.recognize_google.assert_called_once_with("mock_audio", language="en-US")

def test_stt_listen_timeout():
    with patch('voice.speech_to_text.sr.Microphone') as mock_mic:
        stt = SpeechToText()
        stt.recognizer = MagicMock()
        stt.recognizer.listen.side_effect = sr.WaitTimeoutError("Timeout")
        
        result = stt.listen()
        assert result == ""

def test_stt_listen_unknown_value():
    with patch('voice.speech_to_text.sr.Microphone') as mock_mic:
        stt = SpeechToText()
        stt.recognizer = MagicMock()
        stt.recognizer.listen.return_value = "mock_audio"
        stt.recognizer.recognize_google.side_effect = sr.UnknownValueError("Unintelligible")
        
        result = stt.listen()
        assert result == ""

def test_stt_listen_request_error():
    with patch('voice.speech_to_text.sr.Microphone') as mock_mic:
        stt = SpeechToText()
        stt.recognizer = MagicMock()
        stt.recognizer.listen.return_value = "mock_audio"
        stt.recognizer.recognize_google.side_effect = sr.RequestError("API error")
        
        result = stt.listen()
        assert result == ""

def test_stt_close():
    with patch('voice.speech_to_text.sr.Microphone') as mock_mic:
        stt = SpeechToText()
        mock_audio = MagicMock()
        stt.microphone.audio = mock_audio
        stt.close()
        mock_audio.terminate.assert_called_once()
        assert stt.microphone is None
