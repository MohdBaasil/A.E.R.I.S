import logging
import queue
import threading
import time
from typing import Optional
import pyttsx3

logger = logging.getLogger("aeris.voice")

class TextToSpeech:
    """Handles text-to-speech using local pyttsx3 engine."""
    
    def __init__(self, rate: Optional[int] = 175, volume: Optional[float] = 1.0, voice: Optional[str] = None):
        self.rate = rate
        self.volume = volume
        self.voice = voice
        self.last_speak_seconds = 0.0
        self._speech_thread: Optional[threading.Thread] = None
        self._speech_lock = threading.Lock()
        self._speech_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._speaking = threading.Event()
        try:
            self._engine = pyttsx3.init()
            if self._engine and rate is not None:
                try:
                    self._engine.setProperty('rate', rate)
                except Exception as exc:
                    logger.debug("Could not set TTS speech rate: %s", exc)
            if self._engine and volume is not None:
                self._engine.setProperty("volume", max(0.0, min(1.0, volume)))
            if self._engine and voice:
                self._set_preferred_voice(voice)
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}")
            self._engine = None

    def _set_preferred_voice(self, requested_voice: str) -> None:
        """Select a matching installed voice, retaining the default if unavailable."""
        try:
            voices = self._engine.getProperty("voices") or []
            requested = requested_voice.casefold()
            match = next(
                (voice for voice in voices if requested in str(getattr(voice, "id", "")).casefold()
                 or requested in str(getattr(voice, "name", "")).casefold()),
                None,
            )
            if match is None:
                logger.warning("Requested TTS voice '%s' is unavailable; using the system default.", requested_voice)
                return
            self._engine.setProperty("voice", match.id)
        except Exception as exc:
            logger.warning("Could not select TTS voice '%s'; using the system default: %s", requested_voice, exc)

    def speak(self, text: str) -> None:
        """Speaks the given text immediately using the persistent engine."""
        if not text:
            return
        
        if not self._engine:
            logger.warning("TTS engine not initialized. Cannot speak.")
            return

        try:
            started = time.perf_counter()
            self._engine.say(text)
            self._engine.runAndWait()
            self.last_speak_seconds = time.perf_counter() - started
        except Exception as e:
            logger.error(f"Failed to speak text: {e}")

    @property
    def is_speaking(self) -> bool:
        """Whether the optional background utterance is still active."""
        return self._speaking.is_set() or not self._speech_queue.empty()

    def speak_async(self, text: str) -> bool:
        """Queue one utterance on a single worker so a voice loop can accept barge-in."""
        if not text or not self._engine:
            return False
        self._ensure_speech_worker()
        self._speech_queue.put(text)
        return True

    def _ensure_speech_worker(self) -> None:
        if self._speech_thread and self._speech_thread.is_alive():
            return

        def _run() -> None:
            while True:
                text = self._speech_queue.get()
                if text is None:
                    return
                self._speaking.set()
                try:
                    with self._speech_lock:
                        self.speak(text)
                finally:
                    self._speaking.clear()

        self._speech_thread = threading.Thread(target=_run, name="aeris-tts", daemon=True)
        self._speech_thread.start()

    def stop(self) -> None:
        """Stop any pending speech and reset the engine."""
        if not self._engine:
            return
        try:
            self._engine.stop()
        except Exception as e:
            logger.warning(f"Failed to stop TTS engine: {e}")
        while True:
            try:
                self._speech_queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        """Cleanly shutdown TTS engine resources."""
        self.stop()
