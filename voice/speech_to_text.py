import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import speech_recognition as sr

logger = logging.getLogger("aeris.voice.stt")


def get_audio_devices_diagnostics() -> List[Dict[str, Any]]:
    """
    Enumerate all host audio devices and return structured diagnostic information.
    
    Filters and categorizes devices by input/output capability without exposing secrets.
    """
    devices: List[Dict[str, Any]] = []
    pyaudio_module = getattr(sr, "pyaudio", None) or sr.Microphone.get_pyaudio()
    if not pyaudio_module:
        logger.warning("PyAudio is not available for device enumeration.")
        return devices

    pa = None
    try:
        pa = pyaudio_module.PyAudio()

        # Enumerate host APIs to filter discovery to MME, DirectSound, and WASAPI
        supported_host_api_indices: Optional[set[int]] = None
        try:
            host_api_count = pa.get_host_api_count()
            if isinstance(host_api_count, int):
                supported = set()
                for h_idx in range(host_api_count):
                    try:
                        h_info = pa.get_host_api_info_by_index(h_idx)
                        h_name = str(h_info.get("name", "")).upper() if isinstance(h_info, dict) else ""
                        if any(api in h_name for api in ("MME", "DIRECTSOUND", "WASAPI")):
                            supported.add(h_idx)
                    except Exception:
                        pass
                if supported:
                    supported_host_api_indices = supported
        except Exception:
            pass

        device_count = pa.get_device_count()
        for i in range(device_count):
            try:
                info = pa.get_device_info_by_index(i)
            except Exception:
                try:
                    info = pa.get_device_info_by_host_api_device_index(0, i)
                except Exception:
                    continue

            try:
                host_api = int(info.get("hostApi", 0))
            except (ValueError, TypeError):
                host_api = 0

            if supported_host_api_indices is not None and host_api not in supported_host_api_indices:
                continue

            name = str(info.get("name", f"Device {i}")).strip()
            max_input = int(info.get("maxInputChannels", 0))
            max_output = int(info.get("maxOutputChannels", 0))
            sample_rate = float(info.get("defaultSampleRate", 44100.0))
            host_api = int(info.get("hostApi", 0))
            is_input = max_input > 0

            if max_input > 0 and max_output > 0:
                status = "INPUT/OUTPUT"
            elif max_input > 0:
                status = "INPUT"
            else:
                status = "OUTPUT ONLY"

            devices.append({
                "index": i,
                "name": name,
                "max_input_channels": max_input,
                "max_output_channels": max_output,
                "default_sample_rate": sample_rate,
                "host_api": host_api,
                "is_input": is_input,
                "status": status,
            })
    except Exception as exc:
        logger.warning("Error enumerating audio devices via PyAudio: %s", exc)
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass

    return devices


def list_input_devices() -> List[Dict[str, Any]]:
    """Return all detected audio devices that have at least 1 input channel."""
    return [d for d in get_audio_devices_diagnostics() if d.get("is_input", False)]


def print_audio_devices() -> None:
    """Print a clean, formatted listing of audio devices for troubleshooting."""
    devices = get_audio_devices_diagnostics()
    print("\nAERIS AUDIO DEVICES")
    print("===================")
    if not devices:
        print("No audio devices detected.")
        return

    for dev in devices:
        status_tag = f"[{dev['status']}]"
        print(f"[{dev['index']}] {dev['name']}")
        print(f"    Input Channels:  {dev['max_input_channels']}")
        print(f"    Output Channels: {dev['max_output_channels']}")
        print(f"    Sample Rate:     {int(dev['default_sample_rate'])} Hz")
        print(f"    Status:          {status_tag}\n")


def find_microphone_device(
    device_index: Optional[int] = None,
    device_name: Optional[str] = None,
    auto_select: bool = True,
) -> Tuple[Optional[int], Optional[str], Optional[Dict[str, Any]]]:
    """
    Select the most suitable audio input device based on index, name, or auto-selection.
    
    Guarantees that any selected device has max_input_channels > 0 and rejects output-only devices.
    """
    devices = get_audio_devices_diagnostics()
    input_devices = [d for d in devices if d.get("is_input", False)]

    if not input_devices:
        logger.warning("No audio input devices found with positive input channels.")
        return None, None, None

    # 1. Explicit device index
    if device_index is not None:
        match = next((d for d in input_devices if d["index"] == device_index), None)
        if match:
            logger.info("Using configured microphone by index: [%d] %s", match["index"], match["name"])
            return match["index"], match["name"], match
        else:
            logger.warning(
                "Configured MIC_DEVICE_INDEX %s is invalid or not an input device (maxInputChannels > 0). Falling back.",
                device_index,
            )

    # 2. Explicit device name substring
    if device_name and device_name.strip():
        req = device_name.strip().lower()
        match = next((d for d in input_devices if req in d["name"].lower()), None)
        if match:
            logger.info("Using configured microphone by name match: [%d] %s", match["index"], match["name"])
            return match["index"], match["name"], match
        else:
            logger.warning(
                "Configured MIC_DEVICE_NAME '%s' was not found among valid input devices. Falling back.",
                device_name,
            )

    # 3. Auto-selection heuristic (prioritizing physical mics, avoiding virtual/stereo mix/output)
    if auto_select:
        def score_device(dev: Dict[str, Any]) -> int:
            if not dev.get("is_input", False):
                return -1000
            name_lower = dev["name"].lower()
            # Penalize known virtual/loopback/output devices
            if any(bad in name_lower for bad in ["stereo mix", "audiorelay", "virtual mic", "speaker", "headphones", "output"]):
                if "stereo mix" in name_lower or "headphones" in name_lower:
                    return -50
                return 10  # Low fallback priority for virtual mics

            # High priority physical microphone patterns
            if "microphone array" in name_lower or "mic array" in name_lower:
                return 100
            if "realtek" in name_lower and ("mic" in name_lower or "array" in name_lower):
                return 90
            if "microphone" in name_lower or "mic input" in name_lower or "mic" in name_lower:
                return 80
            if "sound mapper - input" in name_lower:
                return 60
            return 40

        scored = [(score_device(d), d) for d in input_devices]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_dev = scored[0]
        if best_score > 0:
            logger.info("Auto-selected microphone: [%d] %s (score=%d)", best_dev["index"], best_dev["name"], best_score)
            return best_dev["index"], best_dev["name"], best_dev

    # 4. Fallback: first available input device
    fallback = input_devices[0]
    logger.info("Falling back to first input device: [%d] %s", fallback["index"], fallback["name"])
    return fallback["index"], fallback["name"], fallback


class SpeechToText:
    """Handles speech-to-text using local SpeechRecognition and PyAudio with safe device selection."""

    def __init__(
        self,
        timeout: int = 4,
        phrase_time_limit: int = 12,
        language: str = "en-US",
        pause_threshold: float = 0.55,
        non_speaking_duration: float = 0.3,
        energy_threshold: int = 300,
        dynamic_energy_threshold: bool = True,
        ambient_duration: float = 0.5,
        auto_calibrate: bool = True,
        device_index: Optional[int] = None,
        device_name: Optional[str] = None,
        auto_select: bool = True,
    ):
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.language = language
        self.pause_threshold = pause_threshold
        self.ambient_duration = ambient_duration
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.non_speaking_duration = non_speaking_duration
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold
        self.last_timings: dict[str, float] = {}
        self.microphone: Optional[sr.Microphone] = None
        self._calibrated = False

        sel_index, sel_name, sel_info = find_microphone_device(
            device_index=device_index,
            device_name=device_name,
            auto_select=auto_select,
        )
        self.device_index = sel_index
        self.device_name = sel_name
        self.device_info = sel_info
        self.is_available = False

        if self.device_index is not None:
            try:
                self.microphone = sr.Microphone(device_index=self.device_index)
                self.is_available = True
                channels = self.device_info.get("max_input_channels", "unknown") if self.device_info else "unknown"
                logger.info(
                    "Initialized SpeechToText with device index %s (%s, channels=%s)",
                    self.device_index,
                    self.device_name,
                    channels,
                )
                if auto_calibrate and self.ambient_duration > 0:
                    self.calibrate_ambient_noise(duration=self.ambient_duration)
            except Exception as e:
                logger.error("Failed to initialize microphone at index %s (%s): %s", self.device_index, self.device_name, e)
                self.microphone = None
                self.is_available = False
        else:
            logger.warning("No usable microphone detected during SpeechToText initialization.")
            self.microphone = None
            self.is_available = False

    def calibrate_ambient_noise(self, duration: Optional[float] = None) -> None:
        """Calibrate recognizer for ambient noise once to eliminate per-turn delay."""
        if not self.microphone:
            return
        cal_duration = duration if duration is not None else self.ambient_duration
        if cal_duration <= 0:
            self._calibrated = True
            return
        try:
            with self.microphone as source:
                logger.info("Calibrating for ambient noise (duration=%.2fs)...", cal_duration)
                self.recognizer.adjust_for_ambient_noise(source, duration=cal_duration)
            self._calibrated = True
            logger.info("Ambient noise calibrated (threshold=%.1f)", getattr(self.recognizer, "energy_threshold", 0))
        except Exception as exc:
            logger.warning("Ambient noise calibration failed or skipped: %s", exc)

    def listen(self, timeout: Optional[float] = None) -> str:
        """Listens to the microphone and returns recognized text."""
        if not self.microphone:
            logger.warning("Microphone not initialized. Cannot listen.")
            return ""

        started = time.perf_counter()
        self.last_timings = {"listening_started": started}
        try:
            with self.microphone as source:
                if not self._calibrated and self.ambient_duration > 0:
                    logger.info("Calibrating ambient noise on first listen...")
                    try:
                        self.recognizer.adjust_for_ambient_noise(source, duration=self.ambient_duration)
                    except Exception as e:
                        logger.warning("Initial noise calibration error: %s", e)
                    self._calibrated = True

                logger.debug("Listening on [%s] %s...", self.device_index, self.device_name)
                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout if timeout is None else timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )

            # SpeechRecognition returns when it has detected the end of a phrase.
            self.last_timings["speech_detected"] = time.perf_counter()

            logger.debug("Recognizing speech...")
            text = self.recognizer.recognize_google(audio, language=self.language)
            self.last_timings["recognition_completed"] = time.perf_counter()
            self.last_timings["stt_seconds"] = self.last_timings["recognition_completed"] - started
            return text.strip()

        except sr.WaitTimeoutError:
            self.last_timings["recognition_completed"] = time.perf_counter()
            logger.debug("Listening timed out while waiting for phrase to start.")
            return ""
        except sr.UnknownValueError:
            self.last_timings["recognition_completed"] = time.perf_counter()
            logger.info("Speech recognition could not understand audio.")
            return ""
        except sr.RequestError as e:
            self.last_timings["recognition_completed"] = time.perf_counter()
            logger.error("Could not request results from Speech Recognition service; %s", e)
            return ""
        except Exception as e:
            self.last_timings["recognition_completed"] = time.perf_counter()
            logger.error("Unexpected error during speech recognition: %s", e)
            raise

    def close(self) -> None:
        """Cleanly release microphone and PyAudio resources."""
        try:
            if self.microphone and hasattr(self.microphone, "audio") and self.microphone.audio:
                self.microphone.audio.terminate()
        except Exception as e:
            logger.debug("Error terminating PyAudio: %s", e)
        self.microphone = None
        self.is_available = False
