import pytest
from unittest.mock import patch, MagicMock
from voice.text_to_speech import TextToSpeech

def test_tts_init_success():
    with patch('voice.text_to_speech.pyttsx3.init') as mock_init:
        tts = TextToSpeech()
        mock_init.assert_called_once()
        assert tts._engine is not None

def test_tts_init_failure():
    with patch('voice.text_to_speech.pyttsx3.init', side_effect=Exception("Init failed")):
        tts = TextToSpeech()
        assert tts._engine is None

def test_tts_speak():
    with patch('voice.text_to_speech.pyttsx3.init') as mock_init:
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        tts = TextToSpeech()
        tts.speak("Hello AERIS")
        
        mock_engine.say.assert_called_once_with("Hello AERIS")
        mock_engine.runAndWait.assert_called_once()

def test_tts_speak_empty():
    with patch('voice.text_to_speech.pyttsx3.init') as mock_init:
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        
        tts = TextToSpeech()
        tts.speak("")
        
        mock_engine.say.assert_not_called()
        mock_engine.runAndWait.assert_not_called()

def test_tts_speak_engine_failed():
    with patch('voice.text_to_speech.pyttsx3.init', side_effect=Exception("Init failed")):
        tts = TextToSpeech()
        tts.speak("Hello")
        # Should not raise exception, just logs a warning

def test_tts_stop():
    with patch('voice.text_to_speech.pyttsx3.init') as mock_init:
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine
        tts = TextToSpeech()
        tts.stop()
        mock_engine.stop.assert_called_once()

def test_tts_stop_no_engine():
    with patch('voice.text_to_speech.pyttsx3.init', side_effect=Exception("Init failed")):
        tts = TextToSpeech()
        tts.stop()  # Should not raise exception
