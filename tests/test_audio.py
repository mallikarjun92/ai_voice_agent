import sounddevice as sd
import numpy as np
import time
from scipy.io.wavfile import write
import os

def test_audio():
    print("--- Audio Device Tester ---")
    devices = sd.query_devices()
    print("\nAvailable Devices:")
    for i, dev in enumerate(devices):
        print(f"[{i}] {dev['name']} (In: {dev['max_input_channels']}, Out: {dev['max_output_channels']})")

    try:
        device_id = input("\nEnter Device ID to test (default 1): ").strip()
        device_id = int(device_id) if device_id else 1
    except ValueError:
        device_id = 1
        print("Invalid input, using ID 1.")

    print(f"\nTesting Device ID: {device_id}")
    
    try:
        # Get device info to avoid sample rate/channel mismatches
        info = sd.query_devices(device_id, 'input')
        fs = int(info['default_samplerate'])
        channels = int(info['max_input_channels'])
        print(f"Device Info: {info['name']}")
        print(f"Using Sample Rate: {fs}, Channels: {channels}")
        
    except Exception as e:
        print(f"Error querying device: {e}")
        print("Hint: The device might be disabled in Windows Sound Settings.")
        return

    duration = 5  # seconds
    print(f"Recording for {duration} seconds... speak or play audio now!")
    
    try:
        # Record with device's native settings
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16', device=device_id)
        sd.wait()
        print("Recording finished.")
        
        # Calculate RMS (average across channels if stereo)
        audio_data = recording.astype(np.float32)
        rms = np.sqrt(np.mean(audio_data**2))
        print(f"RMS Level: {rms:.2f}")
        
        if rms < 100:
            print("WARNING: Very low signal detected.")
            print("1. Check if the device is muted.")
            print("2. Ensure something is actually playing (if testing Stereo Mix).")
        elif rms < 500:
            print("NOTE: Low signal detected.")
        else:
            print("SUCCESS: Audio captured successfully.")
            
        # Save to file (downmix to mono if needed for simplicity)
        test_file = "test_record.wav"
        if channels > 1:
            mono_audio = np.mean(recording, axis=1).astype(np.int16)
        else:
            mono_audio = recording.flatten()
            
        write(test_file, fs, mono_audio)
        print(f"Saved recording to: {os.path.abspath(test_file)}")
        
    except Exception as e:
        print(f"Error opening InputStream: {e}")
        if "Invalid device" in str(e) or "-9996" in str(e):
            print("\nPOSSIBLE FIXES:")
            print("1. Open 'Sound Settings' -> 'More sound settings' (or Control Panel -> Sound).")
            print("2. Go to the 'Recording' tab.")
            print("3. Right-click 'Stereo Mix' and ensure it is 'Enabled'.")
            print("4. Ensure 'Stereo Mix' is set as the default device (optional but helpful).")

if __name__ == "__main__":
    test_audio()
