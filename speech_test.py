import speech_recognition as sr

DEVICE_INDEX = 9

recognizer = sr.Recognizer()

# Don't let ambient-noise calibration set an extremely high threshold.
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = False

print("AERIS Speech Recognition Test")
print("=============================")
print(f"Using microphone device #{DEVICE_INDEX}")
print("Energy threshold:", recognizer.energy_threshold)
print()
print("Speak immediately when you see 'Speak now!'")
print()

try:
    microphone = sr.Microphone(device_index=DEVICE_INDEX)

    with microphone as source:
        print("Speak now!")
        audio = recognizer.listen(
            source,
            timeout=10,
            phrase_time_limit=8
        )

    print("Audio captured!")
    print("Recognizing...")

    text = recognizer.recognize_google(audio)

    print()
    print("You said:", text)

except sr.WaitTimeoutError:
    print("ERROR: No speech detected within 10 seconds.")

except sr.UnknownValueError:
    print("ERROR: Audio was captured, but speech was not understood.")

except sr.RequestError as e:
    print("ERROR: Google Speech Recognition request failed:", e)

except Exception as e:
    print("ERROR:", type(e).__name__, e)

print()
print("Test finished.")