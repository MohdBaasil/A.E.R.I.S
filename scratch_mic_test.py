import pyaudio
import audioop
import time

def test_mics():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    print("--- PyAudio Input Devices ---")
    valid_mics = []
    for i in range(numdevices):
        dev_info = p.get_device_info_by_host_api_device_index(0, i)
        if dev_info.get('maxInputChannels') > 0:
            name = dev_info.get('name')
            print(f"Index {i}: {name} (Channels: {dev_info.get('maxInputChannels')})")
            
            # exclude Stereo Mix, AudioRelay, Speakers
            lower_name = name.lower()
            if "stereo mix" in lower_name or "audiorelay" in lower_name or "speaker" in lower_name or "headphones" in lower_name:
                print("  -> Skipping (likely virtual, stereo mix, or output device)")
                continue
            
            valid_mics.append((i, name, dev_info.get('maxInputChannels'), int(dev_info.get('defaultSampleRate'))))
            
    print("\n--- Testing Valid Mics ---")
    for idx, name, channels, rate in valid_mics:
        print(f"Testing Index {idx}: {name}")
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=idx,
                            frames_per_buffer=1024)
            print(f"  Successfully opened stream for {name}")
            
            print("  Reading audio for 2 seconds. Please speak loudly...")
            start = time.time()
            rms_values = []
            while time.time() - start < 2:
                data = stream.read(1024, exception_on_overflow=False)
                rms = audioop.rms(data, 2)
                rms_values.append(rms)
            
            stream.stop_stream()
            stream.close()
            
            print(f"  RMS values (min/max/avg): {min(rms_values)} / {max(rms_values)} / {sum(rms_values)//len(rms_values)}")
            
        except Exception as e:
            print(f"  Failed to open or read from device {idx}: {e}")

    p.terminate()

if __name__ == "__main__":
    test_mics()
