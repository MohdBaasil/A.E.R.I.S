"""
AERIS entry point — Phase 6 (True Voice-to-Voice Assistant).

Flow:
    Text Mode:  User types -> Assistant -> Gemini -> [Tool Router] -> Response -> Terminal
    Voice Mode: Mic -> STT -> [local cmd check] -> Assistant -> Gemini -> [Tool Router]
                     -> Response -> TTS -> Speaker -> Mic (loop)

Phase 6 additions
-----------------
- Local voice command interception (no round-trip to Gemini for control words).
- TTS speaks every assistant response in voice mode.
- Gemini/network failures are caught and spoken; voice session continues.
- Explicit audio sequencing: mic capture ends before TTS begins.
- Empty / unintelligible speech returns silently to listening without calling Gemini.
- Consecutive-silence counter prevents flooding the log with TTS "I didn't catch that".
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time

from typing import Optional

from ai.provider import AIProvider, ProviderError, create_provider
from config import Config, ConfigError
from core.assistant import Assistant, AI_SERVICE_UNAVAILABLE_MESSAGE
from core.memory import MemoryStore
from core.confirmation import TextConfirmationHandler, VoiceConfirmationHandler
from core.tool_router import ToolRouter
from core.tool_setup import create_tool_registry
from voice.speech_to_text import SpeechToText, get_audio_devices_diagnostics, print_audio_devices
from voice.text_to_speech import TextToSpeech
from voice.activation import WakeWordController, normalize_voice_text
from voice.response_cleanup import prepare_speech_response
from voice.stream_buffer import SentenceBuffer

BANNER = r"""
=================================
        A.E.R.I.S.
Adaptive Electronic Reasoning &
     Intelligence System
=================================
"""

# ---------------------------------------------------------------------------
# Exit / mode-switch vocabularies (checked locally — never sent to Gemini)
# ---------------------------------------------------------------------------

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "shutdown"}
EXIT_PHRASES = {"stop aeris", "stop system", "shut down", "shut down aeris", "stop"}

# Voice-only: commands that switch from voice to text mode
VOICE_TO_TEXT_PHRASES: frozenset[str] = frozenset({
    "deactivate voice",
    "disable voice",
    "stop voice mode",
    "switch to text",
    "switch to text mode",
    "text mode",
    "turn off voice",
    "stop listening",
})
TTS_INTERRUPT_PHRASES: frozenset[str] = frozenset({
    "stop", "stop talking", "quiet", "be quiet", "cancel", "enough",
})


def setup_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, str(log_level).upper(), logging.INFO) if isinstance(log_level, str) else logging.INFO
    logging.basicConfig(
        filename="aeris.log",
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


# ---------------------------------------------------------------------------
# Local command helpers
# ---------------------------------------------------------------------------

def is_exit_command(text: str) -> bool:
    """Return True if *text* matches a recognised exit intent."""
    cleaned = text.strip().lower().strip(".,!?;:")
    if not cleaned:
        return False
    if cleaned in EXIT_COMMANDS or cleaned in EXIT_PHRASES:
        return True
    if any(cleaned.startswith(f"{phrase} ") for phrase in EXIT_PHRASES):
        return True
    words = cleaned.split()
    if words and words[0] in EXIT_COMMANDS:
        return True
    return False


def is_voice_to_text_command(text: str) -> bool:
    """Return True if *text* is a local command to switch voice → text mode."""
    cleaned = text.strip().lower().strip(".,!?;:")
    return cleaned in VOICE_TO_TEXT_PHRASES


def is_tts_interrupt_command(text: str) -> bool:
    """Return True for a local request to stop the current spoken response."""
    return normalize_voice_text(text) in TTS_INTERRUPT_PHRASES


# ---------------------------------------------------------------------------
# TTS helper (never raises — prints fallback on TTS errors)
# ---------------------------------------------------------------------------

def _speak(tts: Optional[TextToSpeech], text: str, logger: logging.Logger) -> None:
    """
    Speak *text* via TTS if available; always print it.

    Does NOT raise. TTS errors are logged and ignored so the voice
    session can continue even if audio output is broken.
    """
    print(f"AERIS: {text}\n")
    if tts:
        try:
            tts.speak(prepare_speech_response(text))
        except Exception as exc:
            logger.error("TTS failed: %s", exc)


def _speak_response_with_barge_in(tts: Optional[TextToSpeech], stt: SpeechToText, text: str, logger: logging.Logger) -> bool:
    """Speak a response and return True when the user interrupted it locally."""
    if not isinstance(tts, TextToSpeech):
        _speak(tts, text, logger)
        return False

    print(f"AERIS: {text}\n")
    if not tts.speak_async(prepare_speech_response(text)):
        return False
    while tts.is_speaking:
        try:
            heard = stt.listen(timeout=1).strip()
        except Exception as exc:
            logger.debug("Barge-in listener unavailable: %s", exc)
            break
        if is_tts_interrupt_command(heard):
            tts.stop()
            print("AERIS: Stopped speaking.\n")
            return True
    return False


def _speak_streamed_response(
    assistant: Assistant,
    tts: Optional[TextToSpeech],
    request: str,
    min_chars: int = 40,
    max_chars: int = 180,
    logger: Optional[logging.Logger] = None,
    stt: Optional[SpeechToText] = None,
    voice_debug: bool = False,
    turn_started: Optional[float] = None,
) -> bool:
    """Send one request through the streaming path, queue natural speech segments, and handle barge-in."""
    import threading
    log = logger or logging.getLogger("aeris.main")
    start_time = turn_started if turn_started is not None else time.perf_counter()
    buffer = SentenceBuffer(min_chars=min_chars, max_chars=max_chars)
    cancel_event = threading.Event()
    received_chunk = False
    interrupted = False

    first_chunk_time: Optional[float] = None
    first_sentence_time: Optional[float] = None
    first_tts_time: Optional[float] = None
    stream_complete_time: Optional[float] = None

    try:
        for chunk in assistant.send_stream(request, cancel_event=cancel_event):
            if cancel_event.is_set():
                break
            if not received_chunk:
                received_chunk = True
                first_chunk_time = time.perf_counter()

            for segment in buffer.feed(chunk):
                if first_sentence_time is None:
                    first_sentence_time = time.perf_counter()
                print(f"AERIS: {segment}\n")
                spoken = prepare_speech_response(segment)
                if spoken:
                    if first_tts_time is None:
                        first_tts_time = time.perf_counter()
                    if isinstance(tts, TextToSpeech):
                        tts.speak_async(spoken)
                    elif tts:
                        tts.speak(spoken)

        stream_complete_time = time.perf_counter()

        if not cancel_event.is_set():
            for segment in buffer.flush():
                if first_sentence_time is None:
                    first_sentence_time = time.perf_counter()
                print(f"AERIS: {segment}\n")
                spoken = prepare_speech_response(segment)
                if spoken:
                    if first_tts_time is None:
                        first_tts_time = time.perf_counter()
                    if isinstance(tts, TextToSpeech):
                        tts.speak_async(spoken)
                    elif tts:
                        tts.speak(spoken)

        if not received_chunk:
            log.warning("Streaming provider returned no text chunks.")

    except ProviderError:
        raise
    except Exception as exc:
        log.error("Error during streaming response: %s", exc)
        raise

    # Handle barge-in while queued TTS is speaking
    if isinstance(tts, TextToSpeech) and stt is not None:
        while tts.is_speaking:
            try:
                heard = stt.listen(timeout=0.5).strip()
            except Exception as exc:
                log.debug("Barge-in listener unavailable: %s", exc)
                break
            if is_tts_interrupt_command(heard):
                tts.stop()
                cancel_event.set()
                interrupted = True
                print("AERIS: Stopped speaking.\n")
                break

    if voice_debug:
        _print_voice_timings(stt, assistant, tts, start_time)
        if first_chunk_time is not None:
            print(f"first_chunk: {first_chunk_time - start_time:.2f}s")
        if first_sentence_time is not None:
            print(f"first_sentence: {first_sentence_time - start_time:.2f}s")

    return interrupted


def _print_voice_timings(stt: object, assistant: Assistant, tts: Optional[TextToSpeech], turn_started: float) -> None:
    """Print compact, non-sensitive latency diagnostics when VOICE_DEBUG is enabled."""
    stt_timings = getattr(stt, "last_timings", {}) or {}
    assistant_timings = getattr(assistant, "last_timings", {}) or {}
    stt_seconds = float(stt_timings.get("stt_seconds") or 0.0)
    llm_seconds = float(assistant_timings.get("llm_seconds") or 0.0)
    tool_seconds = float(assistant_timings.get("tool_seconds") or 0.0)
    tts_seconds = float(getattr(tts, "last_speak_seconds", 0.0) or 0.0) if tts else 0.0
    total_seconds = time.perf_counter() - turn_started
    print("[VOICE DEBUG]")
    print(f"STT: {stt_seconds:.2f}s")
    print(f"LLM: {llm_seconds:.2f}s")
    print(f"Tools: {tool_seconds:.2f}s")
    print(f"TTS: {tts_seconds:.2f}s")
    print(f"Total: {total_seconds:.2f}s")


# ---------------------------------------------------------------------------
# Text mode loop
# ---------------------------------------------------------------------------

def run_text_mode(assistant: Assistant) -> None:
    """Run AERIS in interactive terminal text mode."""
    logger = logging.getLogger("aeris.main")
    print("AERIS: Text mode active.\n")
    logger.info("AERIS running in text mode.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAERIS: Shutting down.")
            break

        if not user_input:
            continue

        if is_exit_command(user_input):
            print("AERIS: Goodbye.")
            logger.info("User requested exit.")
            break

        try:
            reply = assistant.send(user_input)
        except Exception as exc:
            logger.error("Unexpected error in text mode: %s", type(exc).__name__)
            reply = AI_SERVICE_UNAVAILABLE_MESSAGE
        print(f"\nAERIS: {reply}\n")


# ---------------------------------------------------------------------------
# Voice mode loop
# ---------------------------------------------------------------------------

def run_voice_mode(
    assistant: Assistant,
    stt: Optional[SpeechToText] = None,
    tts: Optional[TextToSpeech] = None,
    voice_debug: bool = False,
    wake_word_enabled: bool = False,
    wake_word: str = "aeris",
    active_window_seconds: float = 10.0,
    voice_streaming: bool = False,
    stream_min_chars: int = 40,
    stream_max_chars: int = 180,
) -> None:
    """
    Run AERIS in continuous voice-to-voice conversation mode.

    Sequence per turn
    -----------------
    1. Listen (microphone capture) — mic held only during listening.
    2. Release mic.
    3. Classify input (exit / voice-to-text / normal).
    4. If normal: send to Assistant → Gemini → ToolRouter.
    5. Speak reply via TTS.
    6. Return to step 1.

    All network and TTS errors are caught; the session never crashes.
    """
    logger = logging.getLogger("aeris.main")

    # Initialise STT/TTS if not pre-supplied
    if stt is None:
        try:
            stt = SpeechToText()
        except Exception as exc:
            logger.error("Failed to initialize SpeechToText: %s", exc)
            stt = None

    if tts is None:
        try:
            tts = TextToSpeech()
        except Exception as exc:
            logger.error("Failed to initialize TextToSpeech: %s", exc)
            tts = None

    # Fallback to text mode if no microphone
    if stt is None or getattr(stt, "microphone", None) is None:
        logger.warning("Microphone unavailable for voice mode. Falling back to text mode.")
        print("AERIS: Voice input is unavailable. Switching to text mode.\n")
        if getattr(assistant, "_tool_router", None):
            assistant._tool_router.set_confirmation_handler(TextConfirmationHandler())
        run_text_mode(assistant)
        return

    print("AERIS: Voice mode active.")
    device_name = getattr(stt, "device_name", "Default")
    device_index = getattr(stt, "device_index", None)
    logger.info("Voice mode started — mic=%s (index=%s)", device_name, device_index)

    # Initial greeting (spoken + printed)
    _speak(tts, "AERIS online. Voice mode active.", logger)

    consecutive_errors = 0
    consecutive_silence = 0
    wake_controller = WakeWordController(wake_word_enabled, wake_word, active_window_seconds)
    MAX_CONSECUTIVE_ERRORS = 5
    MAX_SILENCE_TTS = 3  # only say "I didn't catch that" after this many silences

    try:
        while True:
            # ── Step 1: Listen ──────────────────────────────────────────────
            print("\nListening...")
            try:
                user_input = stt.listen().strip()
                consecutive_errors = 0
            except (EOFError, KeyboardInterrupt):
                print("\nAERIS: Shutting down.")
                _speak(tts, "Goodbye.", logger)
                break
            except Exception as exc:
                consecutive_errors += 1
                logger.error(
                    "Voice input error (%d/%d): %s",
                    consecutive_errors, MAX_CONSECUTIVE_ERRORS, exc,
                )
                print(f"AERIS: Voice input error: {exc}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.warning("Too many consecutive voice errors — falling back to text mode.")
                    _speak(tts, "Voice input is unavailable. Switching to text mode.", logger)
                    if getattr(assistant, "_tool_router", None):
                        assistant._tool_router.set_confirmation_handler(TextConfirmationHandler())
                    run_text_mode(assistant)
                    break
                continue

            # ── Step 2: Handle empty / unintelligible speech ────────────────
            if not user_input:
                if wake_word_enabled and not wake_controller.is_active:
                    wake_controller.reset()
                    print("Listening for wake word...")
                    continue
                consecutive_silence += 1
                if consecutive_silence >= MAX_SILENCE_TTS:
                    # Speak once after N silences, then reset counter
                    _speak(tts, "I didn't catch that.", logger)
                    consecutive_silence = 0
                continue
            consecutive_silence = 0

            # ── Step 3: Mic is now released — classify the input ────────────
            print(f"You: {user_input}")
            turn_started = time.perf_counter()

            if is_voice_to_text_command(user_input):
                _speak(tts, "Switching to text mode.", logger)
                logger.info("User switched to text mode via voice command.")
                if getattr(assistant, "_tool_router", None):
                    assistant._tool_router.set_confirmation_handler(TextConfirmationHandler())
                run_text_mode(assistant)
                break

            if is_exit_command(user_input):
                print("\nAERIS: Goodbye.")
                _speak(tts, "Goodbye.", logger)
                logger.info("User requested exit via voice.")
                break

            accepted, request, wake_only = wake_controller.consume(user_input)
            if not accepted:
                logger.debug("Ignoring speech while waiting for wake word.")
                continue
            if wake_only:
                _speak(tts, "Yes?", logger)
                continue

            # ── Step 4: Normal conversation — send to AI pipeline ───────────
            print("AERIS: Thinking...")
            interrupted = False
            if voice_streaming:
                try:
                    interrupted = _speak_streamed_response(
                        assistant=assistant,
                        tts=tts,
                        request=request,
                        min_chars=stream_min_chars,
                        max_chars=stream_max_chars,
                        logger=logger,
                        stt=stt,
                        voice_debug=voice_debug,
                        turn_started=turn_started,
                    )
                except Exception as exc:
                    logger.warning("Streaming voice failed (%s); falling back to non-streaming mode.", exc)
                    try:
                        reply = assistant.send(request)
                        _speak(tts, reply, logger)
                        if voice_debug:
                            _print_voice_timings(stt, assistant, tts, turn_started)
                    except Exception as fallback_exc:
                        logger.error("Fallback non-streaming generate failed: %s", type(fallback_exc).__name__)
                        _speak(tts, AI_SERVICE_UNAVAILABLE_MESSAGE, logger)
                        if voice_debug:
                            _print_voice_timings(stt, assistant, tts, turn_started)
            else:
                try:
                    reply = assistant.send(request)
                    _speak(tts, reply, logger)
                    if voice_debug:
                        _print_voice_timings(stt, assistant, tts, turn_started)
                except Exception as exc:
                    logger.error("Unexpected error from assistant.send(): %s", type(exc).__name__)
                    _speak(tts, AI_SERVICE_UNAVAILABLE_MESSAGE, logger)
                    if voice_debug:
                        _print_voice_timings(stt, assistant, tts, turn_started)

    finally:
        # Always clean up audio resources
        if tts and hasattr(tts, "stop"):
            try:
                tts.stop()
            except Exception as exc:
                logger.debug("Error stopping TTS: %s", exc)
        if stt and hasattr(stt, "close"):
            try:
                stt.close()
            except Exception as exc:
                logger.debug("Error closing STT: %s", exc)


# ---------------------------------------------------------------------------
# Audio diagnostic test
# ---------------------------------------------------------------------------

def run_audio_test(config: Config) -> int:
    """Run a standalone local microphone test and diagnostic report without calling Gemini."""
    print("=================================")
    print("      AERIS AUDIO DIAGNOSTICS    ")
    print("=================================\n")

    print_audio_devices()

    stt_timeout = getattr(config, "stt_timeout", 5)
    stt_phrase_limit = getattr(config, "stt_phrase_time_limit", 10)
    device_index = getattr(config, "mic_device_index", None)
    device_name = getattr(config, "mic_device_name", "")
    auto_select = getattr(config, "mic_auto_select", True)

    print("Initializing SpeechToText for audio test...")
    stt = SpeechToText(
        timeout=stt_timeout,
        phrase_time_limit=stt_phrase_limit,
        device_index=device_index,
        device_name=device_name,
        auto_select=auto_select,
        ambient_duration=getattr(config, "stt_ambient_duration", 0.5),
        energy_threshold=getattr(config, "stt_energy_threshold", 300),
        dynamic_energy_threshold=getattr(config, "stt_dynamic_energy", True),
        pause_threshold=getattr(config, "stt_pause_threshold", 0.55),
    )

    if not stt.microphone:
        print("\n[FAIL] No usable microphone detected.")
        print("Please connect a working microphone or configure MIC_DEVICE_INDEX / MIC_DEVICE_NAME in .env.")
        return 1

    print(f"\n[OK] Selected Microphone: [{stt.device_index}] {stt.device_name}")
    if stt.device_info:
        print(f"     Input Channels: {stt.device_info.get('max_input_channels')}")
        print(f"     Sample Rate:    {int(stt.device_info.get('default_sample_rate', 44100))} Hz")

    print("\nListening for 5 seconds. Please speak into the microphone now...")
    try:
        text = stt.listen(timeout=5)
        if text:
            print(f"\n[SUCCESS] Speech detected and recognised: \"{text}\"")
        else:
            print("\n[INFO] No speech detected or speech was silent during the test window.")
            print("Microphone stream opened successfully with no hardware errors.")
    except Exception as exc:
        print(f"\n[ERROR] Audio capture failed: {exc}")
        return 1
    finally:
        stt.close()

    print("\nAudio diagnostic test completed.")
    return 0


# ---------------------------------------------------------------------------
# Provider diagnostic test
# ---------------------------------------------------------------------------

def run_provider_test(config: Config) -> int:
    """Test AI provider connectivity without running the full assistant or voice pipeline."""
    api_key = getattr(config, "ai_api_key", "")
    model = getattr(config, "ai_model", "")
    connect_timeout = getattr(config, "ai_connect_timeout", 5)
    read_timeout = getattr(config, "ai_read_timeout", 12)
    retries = getattr(config, "ai_retries", getattr(config, "ai_max_retries", 1))
    retry_backoff = getattr(config, "ai_retry_backoff", 0.5)
    provider_name = "Ollama" if getattr(config, "ai_provider", "gemini") == "ollama" else "Gemini"

    print("AERIS Provider Test")
    print(f"Provider: {provider_name}")
    print(f"Model: {model}")
    if provider_name == "Gemini":
        print(f"API key: {'configured' if api_key else 'NOT SET'}")
        print(f"Timeouts: {connect_timeout}s connect, {read_timeout}s read")
        print(f"Retries: up to {retries}, exponential backoff from {retry_backoff:g}s")
    else:
        print("API key: not required")
        print(f"Local request timeout: {getattr(config, 'ollama_timeout', 120)}s")

    try:
        provider = create_provider(config)
    except Exception as exc:
        print("Status: FAILED")
        print(f"Reason: Configuration error: {type(exc).__name__}")
        return 1

    max_wait = getattr(provider, "max_wait_seconds", None)
    if callable(max_wait) and isinstance(config, Config):
        budget_seconds = max_wait()
        if isinstance(budget_seconds, (int, float)):
            print(f"No-response timeout budget: {budget_seconds:.1f}s (connect/read timeouts plus retry backoff)")

    health = provider.health()
    status = health.get("status", "UNKNOWN")
    latency = health.get("latency", None)
    reason = health.get("reason", "")

    if status == "HEALTHY":
        print("Status: OK")
        if latency is not None:
            print(f"Latency: {latency:.2f}s")
        return 0
    if status == "DEGRADED":
        print("Status: DEGRADED")
        if reason:
            print(f"Reason: {reason}")
        if latency is not None:
            print(f"Latency: {latency:.2f}s")
        return 0

    print("Status: FAILED")
    if reason:
        print(f"Reason: {reason}")
    if latency is not None:
        print(f"Latency: {latency:.2f}s")
    return 1


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

def main() -> int:
    setup_logging()
    logger = logging.getLogger("aeris.main")

    print(BANNER)

    try:
        config = Config.load()
    except ConfigError as exc:
        print(f"AERIS: Startup failed - {exc}")
        logger.error("Startup failed: %s", exc)
        return 1

    setup_logging(getattr(config, "log_level", "INFO"))

    parser = argparse.ArgumentParser(description="AERIS AI Assistant")
    parser.add_argument("--voice", "-v", action="store_true", help="Start directly in voice mode")
    parser.add_argument("--text", "-t", action="store_true", help="Start directly in text mode")
    parser.add_argument("--audio-test", action="store_true", help="Run standalone local microphone diagnostic test")
    parser.add_argument("--devices", "--list-mics", dest="list_mics", action="store_true", help="List all detected audio devices and exit")
    parser.add_argument("--provider-test", action="store_true", help="Test AI provider connectivity and exit")
    args, _ = parser.parse_known_args()

    if getattr(args, "list_mics", False):
        print_audio_devices()
        return 0

    if args.audio_test:
        return run_audio_test(config)

    if args.provider_test:
        return run_provider_test(config)

    try:
        provider = create_provider(config)
    except ProviderError as exc:
        print(f"AERIS: Startup failed - {exc}")
        logger.error("Startup failed: %s", exc)
        return 1

    if isinstance(provider, AIProvider) and provider.requires_startup_health_check:
        health = provider.health()
        if health.get("status") != "HEALTHY":
            message = health.get("reason") or "Ollama is unavailable. Make sure Ollama is running."
            print(f"AERIS: {message}")
            logger.warning("Selected AI provider failed startup health check: Ollama")
            return 1

    if getattr(config, "ollama_warmup", True) and hasattr(provider, "warmup"):
        try:
            provider.warmup()
        except Exception as exc:
            logger.warning("Provider warmup failed gracefully: %s", exc)

    # Initialise tool system
    registry = create_tool_registry()
    logger.info("Tool registry initialised with %d tools.", len(registry.list_names()))
    memory_path = getattr(
        config,
        "memory_db_path",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "aeris_memory.sqlite3"),
    )
    if not isinstance(memory_path, (str, os.PathLike)):
        memory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "aeris_memory.sqlite3")
    memory_max_records = getattr(config, "memory_max_records", 500)
    if not isinstance(memory_max_records, int) or isinstance(memory_max_records, bool):
        memory_max_records = 500
    try:
        memory_store = MemoryStore(memory_path, max_records=memory_max_records)
    except (OSError, sqlite3.Error) as exc:
        logger.error("Persistent memory could not be initialized (%s)", type(exc).__name__)
        memory_store = None

    if args.voice:
        mode = "2"
    elif args.text:
        mode = "1"
    else:
        print("Select interaction mode:")
        print("  [1] Text mode")
        print("  [2] Voice mode")
        try:
            mode = input("Enter choice (1/2, default: 1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAERIS: Shutting down.")
            return 0

    stt_timeout = getattr(config, "stt_timeout", 4)
    stt_timeout = stt_timeout if isinstance(stt_timeout, int) else 4
    stt_phrase_limit = getattr(config, "stt_phrase_time_limit", 12)
    stt_phrase_limit = stt_phrase_limit if isinstance(stt_phrase_limit, int) else 12
    max_history = getattr(config, "max_history_messages", 30)
    voice_max_history = getattr(config, "voice_max_history_messages", max_history)
    max_rounds = getattr(config, "max_tool_rounds", 5)

    if mode == "2":
        try:
            stt = SpeechToText(
                timeout=stt_timeout,
                phrase_time_limit=stt_phrase_limit,
                pause_threshold=getattr(config, "stt_pause_threshold", 0.55),
                non_speaking_duration=getattr(config, "stt_non_speaking_duration", 0.3),
                energy_threshold=getattr(config, "stt_energy_threshold", 300),
                dynamic_energy_threshold=getattr(config, "stt_dynamic_energy", True),
                ambient_duration=getattr(config, "stt_ambient_duration", 0.5),
                device_index=getattr(config, "mic_device_index", None),
                device_name=getattr(config, "mic_device_name", ""),
                auto_select=getattr(config, "mic_auto_select", True),
            )
        except Exception as exc:
            logger.error("Failed to initialize SpeechToText: %s", exc)
            stt = None

        try:
            tts = TextToSpeech(
                rate=getattr(config, "tts_rate", 175),
                volume=getattr(config, "tts_volume", 1.0),
                voice=getattr(config, "tts_voice", ""),
            )
        except Exception as exc:
            logger.error("Failed to initialize TextToSpeech: %s", exc)
            tts = None

        if not stt or not getattr(stt, "microphone", None):
            print("AERIS: No usable microphone detected.")
            print("AERIS: Switching to text mode.\n")
            logger.warning("Microphone unavailable, falling back to text mode.")
            confirmation = TextConfirmationHandler()
            tool_router = ToolRouter(registry, confirmation_handler=confirmation)
            assistant = Assistant(
                provider,
                tool_router=tool_router,
                max_history_messages=max_history,
                max_tool_rounds=max_rounds,
                memory_store=memory_store,
            )
            print("AERIS: Online.\n")
            run_text_mode(assistant)
        else:
            confirmation = VoiceConfirmationHandler(tts=tts, stt=stt)
            tool_router = ToolRouter(registry, confirmation_handler=confirmation)
            assistant = Assistant(
                provider,
                tool_router=tool_router,
                max_history_messages=voice_max_history,
                max_tool_rounds=max_rounds,
                memory_store=memory_store,
            )
            print("AERIS: Online.\n")
            run_voice_mode(
                assistant,
                stt,
                tts,
                voice_debug=getattr(config, "voice_debug", False),
                wake_word_enabled=getattr(config, "wake_word_enabled", False),
                wake_word=getattr(config, "wake_word", "aeris"),
                active_window_seconds=getattr(config, "voice_active_window", 10.0),
                voice_streaming=getattr(config, "voice_streaming", False),
                stream_min_chars=getattr(config, "voice_stream_min_chars", 40),
                stream_max_chars=getattr(config, "voice_stream_max_chars", 180),
            )
    else:
        confirmation = TextConfirmationHandler()
        tool_router = ToolRouter(registry, confirmation_handler=confirmation)
        assistant = Assistant(
            provider,
            tool_router=tool_router,
            max_history_messages=max_history,
            max_tool_rounds=max_rounds,
            memory_store=memory_store,
        )
        print("AERIS: Online.\n")
        logger.info("AERIS started with provider=%s model=%s", config.ai_provider, config.ai_model)
        run_text_mode(assistant)

    return 0


if __name__ == "__main__":
    sys.exit(main())
