import sys
import os
import logging

# Set up simple logging for the manual test
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure the root path is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.speech_to_text import SpeechToText

def main():
    print("Testing AERIS speech-to-text...")
    print("Initializing microphone...")
    stt = SpeechToText(timeout=10, phrase_time_limit=15)
    
    if stt.microphone:
        print("\nReady! Speak something...")
        text = stt.listen()
        if text:
            print(f"\nRecognized text: '{text}'")
        else:
            print("\nNo text was recognized (timeout, silence, or error).")
    else:
        print("Failed to initialize microphone. Check your hardware.")

if __name__ == "__main__":
    main()
