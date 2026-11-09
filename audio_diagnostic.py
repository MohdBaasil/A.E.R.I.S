import pyaudio

p = pyaudio.PyAudio()

print("=== AUDIO HOST APIS ===")
for i in range(p.get_host_api_count()):
    info = p.get_host_api_info_by_index(i)
    print(i, info["name"])

print()
print("=== INPUT DEVICES ===")

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)

    if info["maxInputChannels"] > 0:
        print(
            f"Device {i}: {info['name']} | "
            f"Host API: {info['hostApi']} | "
            f"Channels: {info['maxInputChannels']} | "
            f"Rate: {info['defaultSampleRate']}"
        )

print()
print("=== TESTING INPUT DEVICES ===")

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)

    if info["maxInputChannels"] <= 0:
        continue

    print(f"\nTesting device {i}: {info['name']}")

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=int(info["defaultSampleRate"]),
            input=True,
            input_device_index=i,
            frames_per_buffer=1024,
        )

        print("  SUCCESS: device opened")

        stream.read(1024, exception_on_overflow=False)

        print("  SUCCESS: audio frame captured")

        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

p.terminate()

print("\nDiagnostic finished.")
