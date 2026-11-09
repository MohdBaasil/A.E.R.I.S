from unittest.mock import MagicMock, patch
import pytest

from config import Config
from main import main, run_audio_test, run_voice_mode
from voice.speech_to_text import (
    SpeechToText,
    find_microphone_device,
    get_audio_devices_diagnostics,
    list_input_devices,
    print_audio_devices,
)


# ============================================================================
# 1. Device Discovery & Diagnostic Tests
# ============================================================================

def _mock_pyaudio_devices(device_list):
    """Helper to mock PyAudio device enumeration."""
    mock_pa_module = MagicMock()
    mock_pa_instance = MagicMock()
    mock_pa_module.PyAudio.return_value = mock_pa_instance
    mock_pa_instance.get_device_count.return_value = len(device_list)

    def get_info(host_idx, dev_idx):
        return device_list[dev_idx]

    mock_pa_instance.get_device_info_by_host_api_device_index.side_effect = get_info
    mock_pa_instance.get_device_info_by_index.side_effect = lambda idx: device_list[idx]
    return mock_pa_module


class TestAudioDeviceDiscovery:
    def test_get_audio_devices_diagnostics_categorizes_devices(self, monkeypatch):
        devices = [
            {"name": "Microsoft Sound Mapper - Input", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 44100.0, "hostApi": 0},
            {"name": "Headphones (Realtek HD Audio output)", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 48000.0, "hostApi": 0},
            {"name": "Microphone Array (Realtek)", "maxInputChannels": 4, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
            {"name": "Headset Hands-Free Audio", "maxInputChannels": 1, "maxOutputChannels": 1, "defaultSampleRate": 16000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        diagnostics = get_audio_devices_diagnostics()
        assert len(diagnostics) == 4
        assert diagnostics[0]["status"] == "INPUT"
        assert diagnostics[1]["status"] == "OUTPUT ONLY"
        assert diagnostics[1]["is_input"] is False
        assert diagnostics[2]["status"] == "INPUT"
        assert diagnostics[2]["is_input"] is True
        assert diagnostics[3]["status"] == "INPUT/OUTPUT"

    def test_list_input_devices_filters_out_output_only_devices(self, monkeypatch):
        devices = [
            {"name": "Speakers (Realtek)", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 48000.0, "hostApi": 0},
            {"name": "Microphone Array", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
            {"name": "HDMI Output", "maxInputChannels": 0, "maxOutputChannels": 6, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        inputs = list_input_devices()
        assert len(inputs) == 1
        assert inputs[0]["name"] == "Microphone Array"
        assert inputs[0]["index"] == 1

    def test_print_audio_devices_output(self, monkeypatch, capsys):
        devices = [
            {"name": "Microphone Array", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        print_audio_devices()
        captured = capsys.readouterr().out
        assert "AERIS AUDIO DEVICES" in captured
        assert "Microphone Array" in captured
        assert "Input Channels:  2" in captured
        assert "[INPUT]" in captured


# ============================================================================
# 2. Microphone Selection Logic Tests
# ============================================================================

class TestMicrophoneSelectionLogic:
    def test_auto_select_prefers_microphone_array_over_output_and_virtual(self, monkeypatch):
        devices = [
            {"name": "Microsoft Sound Mapper - Input", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 44100.0, "hostApi": 0},
            {"name": "Virtual Mic (AudioRelay)", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
            {"name": "Headphones (Realtek 2nd output)", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 48000.0, "hostApi": 0},
            {"name": "Microphone Array (Realtek Audio)", "maxInputChannels": 4, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
            {"name": "Stereo Mix (Realtek)", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        index, name, info = find_microphone_device(auto_select=True)
        assert index == 3
        assert name == "Microphone Array (Realtek Audio)"
        assert info["max_input_channels"] == 4

    def test_select_by_configured_index(self, monkeypatch):
        devices = [
            {"name": "Mic 0", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 44100.0, "hostApi": 0},
            {"name": "Mic 1", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        index, name, _ = find_microphone_device(device_index=1)
        assert index == 1
        assert name == "Mic 1"

    def test_configured_index_on_output_only_device_falls_back(self, monkeypatch):
        devices = [
            {"name": "Speakers (Output Only)", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 48000.0, "hostApi": 0},
            {"name": "Real Microphone", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        # Configured index 0 is output-only (channels=0); must fallback to Real Microphone
        index, name, _ = find_microphone_device(device_index=0, auto_select=True)
        assert index == 1
        assert name == "Real Microphone"

    def test_select_by_configured_name_substring(self, monkeypatch):
        devices = [
            {"name": "Built-in Mic", "maxInputChannels": 1, "maxOutputChannels": 0, "defaultSampleRate": 44100.0, "hostApi": 0},
            {"name": "USB Wireless Headset Mic", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        index, name, _ = find_microphone_device(device_name="wireless")
        assert index == 1
        assert name == "USB Wireless Headset Mic"

    def test_no_input_devices_returns_none(self, monkeypatch):
        devices = [
            {"name": "Speakers", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 48000.0, "hostApi": 0},
            {"name": "Headphones", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        index, name, info = find_microphone_device()
        assert index is None
        assert name is None
        assert info is None


# ============================================================================
# 3. SpeechToText Initialization & Lifecycle Tests
# ============================================================================

class TestSpeechToTextLifecycle:
    def test_stt_initialization_with_no_devices_does_not_crash(self, monkeypatch):
        devices = [
            {"name": "Speakers", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        stt = SpeechToText()
        assert stt.microphone is None
        assert stt.is_available is False
        assert stt.device_index is None
        assert stt.listen() == ""
        # Calibrate should not raise
        stt.calibrate_ambient_noise()
        stt.close()

    def test_stt_initialization_success_sets_microphone(self, monkeypatch):
        devices = [
            {"name": "Microphone Array", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        with patch("speech_recognition.Microphone", return_value=MagicMock()):
            stt = SpeechToText(auto_calibrate=False)
            assert stt.microphone is not None
            assert stt.is_available is True
            assert stt.device_index == 0


# ============================================================================
# 4. Error Distinguishing & Audio Test CLI
# ============================================================================

class TestFailureHandlingAndCLIDiagnostics:
    def test_audio_test_mode_reports_failure_when_no_mic(self, monkeypatch, capsys):
        devices = [
            {"name": "Speakers", "maxInputChannels": 0, "maxOutputChannels": 2, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        config = Config(
            ai_provider="gemini",
            ai_api_key="test",
            ai_model="test",
        )
        exit_code = run_audio_test(config)
        assert exit_code == 1
        captured = capsys.readouterr().out
        assert "No usable microphone detected" in captured

    def test_audio_test_mode_success_path(self, monkeypatch, capsys):
        devices = [
            {"name": "Microphone Array", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)

        mock_mic = MagicMock()
        with patch("speech_recognition.Microphone", return_value=mock_mic), \
             patch("voice.speech_to_text.SpeechToText.listen", return_value="test voice"):
            config = Config(
                ai_provider="gemini",
                ai_api_key="test",
                ai_model="test",
            )
            exit_code = run_audio_test(config)
            assert exit_code == 0
            captured = capsys.readouterr().out
            assert "[SUCCESS] Speech detected and recognised: \"test voice\"" in captured

    def test_main_cli_devices_flag_exits_cleanly(self, monkeypatch, capsys):
        devices = [
            {"name": "Microphone Array", "maxInputChannels": 2, "maxOutputChannels": 0, "defaultSampleRate": 48000.0, "hostApi": 0},
        ]
        mock_pa = _mock_pyaudio_devices(devices)
        monkeypatch.setattr("speech_recognition.Microphone.get_pyaudio", lambda: mock_pa)
        monkeypatch.setattr("sys.argv", ["main.py", "--devices"])

        exit_code = main()
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "AERIS AUDIO DEVICES" in captured


# ============================================================================
# 5. Configuration Loading for Microphone
# ============================================================================

class TestMicrophoneConfiguration:
    def test_config_loads_microphone_options(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "gemini")
        monkeypatch.setenv("AI_API_KEY", "test-key")
        monkeypatch.setenv("AI_MODEL", "gemini-3.5-flash-lite")
        monkeypatch.setenv("MIC_DEVICE_INDEX", "3")
        monkeypatch.setenv("MIC_DEVICE_NAME", "Realtek Array")
        monkeypatch.setenv("MIC_AUTO_SELECT", "false")

        config = Config.load()
        assert config.mic_device_index == 3
        assert config.mic_device_name == "Realtek Array"
        assert config.mic_auto_select is False
