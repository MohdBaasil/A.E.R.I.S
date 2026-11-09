import sys
import os
import time
import speech_recognition as sr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice.speech_to_text import SpeechToText

def run_verification():
    print("=" * 60)
    print("       AERIS STT REAL-WORLD VERIFICATION TEST")
    print("=" * 60)
    
    stt = SpeechToText(timeout=10, phrase_time_limit=12)
    
    print(f"\n[1] Selected Microphone Device:")
    print(f"    - Index : {stt.device_index}")
    print(f"    - Name  : {stt.device_name}")
    
    if not stt.microphone:
        print("\n[ERROR] Microphone failed to initialize!")
        print("Hardware / PyAudio initialization failure.")
        return
        
    print(f"\n[2] Listening Configuration:")
    print(f"    - Timeout (waiting for speech) : {stt.timeout}s")
    print(f"    - Phrase Duration Limit        : {stt.phrase_time_limit}s")
    print(f"    - Language                     : {stt.language}")
    
    print("\n" + "-" * 60)
    print(">>> PLEASE SPEAK NOW: 'Hello AERIS, this is a microphone test.' <<<")
    print("-" * 60)
    
    print("\n[3] Adjusting for ambient noise (0.5s)...")
    try:
        with stt.microphone as source:
            stt.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("[4] Status: LISTENING... (Speak clearly into your microphone)")
            audio = stt.recognizer.listen(
                source,
                timeout=stt.timeout,
                phrase_time_limit=stt.phrase_time_limit
            )
            print("[5] Status: AUDIO CAPTURED. Contacting Google Speech Recognition API...")
            
            try:
                text = stt.recognizer.recognize_google(audio, language=stt.language)
                print("\n" + "=" * 60)
                print(f"[SUCCESS] Recognized Text: '{text}'")
                print("=" * 60)
                if "hello" in text.lower() or "aeris" in text.lower() or "test" in text.lower() or "microphone" in text.lower():
                    print("Match Evaluation: Sentence successfully detected and transcribed!")
                else:
                    print("Sentence captured, but wording differed slightly.")
            except sr.UnknownValueError:
                print("\n[RECOGNITION FAILURE - UnknownValueError]")
                print("Audio frames were captured successfully by the microphone, but the audio was unintelligible or silent (no distinct words recognized).")
                print("Note: Microphone hardware/stream is working; audio was just not understood.")
            except sr.RequestError as e:
                print(f"\n[API FAILURE - RequestError]")
                print(f"Could not reach Google Speech Recognition service: {e}")
                print("Check internet connection.")
                
    except sr.WaitTimeoutError:
        print("\n[TIMEOUT FAILURE - WaitTimeoutError]")
        print("Microphone opened successfully, but no speech was detected before timeout expired.")
    except Exception as e:
        print(f"\n[PYAUDIO / SYSTEM FAILURE] Exception occurred: {type(e).__name__}: {e}")

if __name__ == "__main__":
    run_verification()
