import speech_recognition as sr

def test_sr():
    r = sr.Recognizer()
    print("Listing microphones through SpeechRecognition:")
    mics = sr.Microphone.list_microphone_names()
    for i, mic_name in enumerate(mics):
        print(f"Index {i}: {mic_name}")

    print("\nAttempting to use default microphone...")
    try:
        with sr.Microphone() as source:
            print("Successfully opened default microphone.")
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Listening for 3 seconds...")
            try:
                audio = r.listen(source, timeout=3, phrase_time_limit=3)
                print("Audio captured. Attempting recognition...")
                text = r.recognize_google(audio)
                print(f"Recognized: {text}")
            except sr.WaitTimeoutError:
                print("Timeout: No speech detected.")
            except sr.UnknownValueError:
                print("Speech unintelligible.")
            except Exception as e:
                print(f"Error during recognition: {e}")
    except Exception as e:
        print(f"Failed to open default microphone: {e}")

if __name__ == "__main__":
    test_sr()
