import sys
import os

# Ensure the root path is in sys.path so we can import from voice module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.text_to_speech import TextToSpeech

def main():
    print("Testing AERIS text-to-speech...")
    tts = TextToSpeech()
    if tts._engine:
        text = "Hello. I am AERIS. My voice initialization is complete."
        print(f"Speaking: {text}")
        tts.speak(text)
        print("Done.")
    else:
        print("Failed to initialize TTS engine.")

if __name__ == "__main__":
    main()
