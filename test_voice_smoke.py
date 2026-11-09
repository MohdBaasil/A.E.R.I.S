import sys
import os

from config import Config
from ai.provider import create_provider
from core.assistant import Assistant
from voice.text_to_speech import TextToSpeech
from voice.speech_to_text import SpeechToText

def run_smoke_test():
    print("=" * 60)
    print("   AERIS Phase 2 Voice Integration Smoke Test")
    print("=" * 60)

    # 1. Config & Provider
    print("\n[1] Loading Configuration...")
    config = Config.load()
    print(f"    - Provider: {config.ai_provider}")
    print(f"    - Model   : {config.ai_model}")

    print("\n[2] Initializing Gemini AI Provider...")
    provider = create_provider(config)
    assistant = Assistant(provider)
    print("    - Assistant initialized successfully.")

    # 2. Text-to-Speech Engine
    print("\n[3] Initializing TextToSpeech Engine...")
    tts = TextToSpeech()
    if tts._engine:
        print("    - TTS engine ready.")
    else:
        print("    - TTS engine failed to initialize.")
        return False

    # 3. Speech-to-Text Engine
    print("\n[4] Initializing SpeechToText Engine...")
    stt = SpeechToText()
    if stt.microphone:
        print(f"    - STT ready on device {stt.device_index}: {stt.device_name}")
    else:
        print("    - STT microphone failed.")
        return False

    # 4. Live Gemini + TTS roundtrip test
    test_prompt = "Hello AERIS, reply in 5 words or less to confirm you are online."
    print(f"\n[5] Sending test message to Gemini: '{test_prompt}'")
    reply = assistant.send(test_prompt)
    print(f"    - Gemini Reply: '{reply}'")

    print("\n[6] Speaking Gemini response through TTS...")
    tts.speak(reply)
    print("    - TTS output complete.")

    print("\n" + "=" * 60)
    print("   ALL INTEGRATION COMPONENTS VERIFIED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
