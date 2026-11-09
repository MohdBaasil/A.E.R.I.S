import pyaudio
import audioop

DEVICE_INDEX = 9
RATE = 48000
CHUNK = 1024

p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK,
)

print("Microphone level test")
print("=====================")
print("Speak normally for 5 seconds.")
print("Press Ctrl+C when finished.")
print()

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        rms = audioop.rms(data, 2)

        bars = min(int(rms / 100), 50)

        print(
            f"RMS: {rms:6d} | "
            + "#" * bars,
            end="\r",
            flush=True,
        )

except KeyboardInterrupt:
    print("\n\nTest stopped.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
