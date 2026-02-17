import sounddevice as sd
import numpy as np
import time
from scipy.io.wavfile import write
from voice_agent.utils.logger import log
from voice_agent.config import INPUT_AUDIO, SILENCE_THRESHOLD, SAMPLERATE

class AudioHandler:
    def __init__(self, samplerate=16000):
        self.fs = samplerate
        self.input_device = None
        self.output_device = None

    def list_devices(self):
        devices = sd.query_devices()
        log("AUDIO", "Available Devices:")
        for i, dev in enumerate(devices):
            log("AUDIO", f"[{i}] {dev['name']} (In: {dev['max_input_channels']}, Out: {dev['max_output_channels']})")

    def set_input_device(self, device_id):
        self.input_device = device_id
        log("AUDIO", f"Input device set to: {device_id}")

    def record(self, session, max_record_time=10.0, silence_limit=0.8, initial_timeout=3.0):
        log("REC", "Start" if not session.is_speaking else "LISTENING (DURING SPEECH)")
        recorded_chunks = []
        
        # VAD State
        last_voice_at = time.time()
        voice_detected = False
        
        def callback(indata, frames, time_info, status):
            nonlocal last_voice_at, voice_detected
            # Calculate RMS to detect voice
            data = np.frombuffer(indata, dtype=np.int16)
            rms = np.sqrt(np.mean(data.astype(np.float32)**2))
            
            if rms > 1500: # Threshold for "active" voice
                if session.is_speaking or session.is_generating:
                    session.abort()
                last_voice_at = time.time()
                voice_detected = True
                
            recorded_chunks.append(indata[:])

        try:
            with sd.RawInputStream(samplerate=self.fs, channels=1, dtype='int16', 
                                  device=self.input_device, callback=callback):
                start_time = time.time()
                while time.time() - start_time < max_record_time:
                    time.sleep(0.05)
                    curr_time = time.time()
                    
                    if voice_detected:
                        if (curr_time - last_voice_at) > silence_limit:
                            break
                    else:
                        if (curr_time - start_time) > initial_timeout:
                            break
        except Exception as e:
            log("AUDIO", f"Recording error: {e}")
            return False

        if not recorded_chunks: return False
        audio = np.frombuffer(b''.join(recorded_chunks), dtype=np.int16)
        
        rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
        if rms < SILENCE_THRESHOLD: return False

        write(INPUT_AUDIO, self.fs, audio)
        log("REC", "End")
        return True

    def create_output_stream(self, samplerate=SAMPLERATE):
        stream = sd.RawOutputStream(
            samplerate=samplerate,
            channels=1,
            dtype='int16',
            blocksize=4096
        )
        stream.start()
        return stream
