import pyaudio

DEVICE_INDEX = 22

p = pyaudio.PyAudio()

try:
    info = p.get_device_info_by_index(DEVICE_INDEX)

    print("Device:")
    print(info)
    print()
    print("Opening microphone...")

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=48000,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=1024,
    )

    print("Microphone opened successfully!")
    print("Speak for 3 seconds...")

    for _ in range(int(48000 / 1024 * 3)):
        stream.read(1024, exception_on_overflow=False)

    print("Audio was captured successfully.")

    stream.stop_stream()
    stream.close()

except Exception as e:
    print()
    print("MICROPHONE ERROR:")
    print(type(e).__name__, e)

finally:
    p.terminate()

print("Test finished.")
