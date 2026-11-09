import speech_recognition as sr

recognizer = sr.Recognizer()

print("AERIS Microphone Test")
print("=====================")
print("Using microphone device #22")
print("Adjusting for ambient noise...")

microphone = sr.Microphone(device_index=22)

with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)

    print("Energy threshold:", recognizer.energy_threshold)
    print()
    print("Speak now!")
    print("Say: Hello AERIS")
    print()

    try:
        audio = recognizer.listen(
            source,
            timeout=10,
            phrase_time_limit=10
        )

        print("Audio captured successfully!")
        print("Recognizing speech...")

        text = recognizer.recognize_google(audio)

        print()
        print("You said:", text)

    except sr.WaitTimeoutError:
        print("ERROR: No speech detected within 10 seconds.")

    except sr.UnknownValueError:
        print("ERROR: Speech was detected, but Google could not understand it.")

    except sr.RequestError as e:
        print("ERROR: Google Speech Recognition request failed:", e)

    except Exception as e:
        print("ERROR:", type(e).__name__, e)

print()
print("Microphone test finished.")