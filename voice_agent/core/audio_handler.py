import sounddevice as sd
import numpy as np
import time
import queue
from scipy.io.wavfile import write
from voice_agent.utils.logger import log
from voice_agent.config import INPUT_AUDIO, SILENCE_THRESHOLD, SAMPLERATE, VAD_THRESHOLD

class AudioHandler:
    def __init__(self, samplerate=None):
        self.fs = samplerate
        self.channels = 1
        self.input_device = None
        self.output_device = None
        self.speech_start_time = 0
        self.out_queue = queue.Queue()
        self.in_queue = queue.Queue()
        self.input_stream = None
        self.input_info = None

    def list_devices(self):
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        log("AUDIO", "Available Devices:")
        for i, dev in enumerate(devices):
            host_api = host_apis[dev['hostapi']]['name']
            log("AUDIO", f"[{i}] {dev['name']} ({host_api}) (In: {dev['max_input_channels']}, Out: {dev['max_output_channels']})")

    def set_input_device(self, device_id):
        try:
            info = sd.query_devices(device_id, 'input')
            if info['max_input_channels'] > 0:
                self.input_device = device_id
                log("AUDIO", f"Input device set to: {device_id} ({info['name']})")
            else:
                log("AUDIO", f"ERROR: Device {device_id} has no input channels!")
        except Exception as e:
            log("AUDIO", f"ERROR: Invalid input device {device_id}: {e}")

    def set_output_device(self, device_id):
        try:
            info = sd.query_devices(device_id, 'output')
            if info['max_output_channels'] > 0:
                self.output_device = device_id
                log("AUDIO", f"Output device set to: {device_id} ({info['name']})")
            else:
                log("AUDIO", f"ERROR: Device {device_id} has no output channels!")
        except Exception as e:
            log("AUDIO", f"ERROR: Invalid output device {device_id}: {e}")

    def start_background_recording(self):
        """Starts a persistent input stream to avoid hardware toggle stutters."""
        if self.input_stream and self.input_stream.active:
            return

        try:
            # Query and cache device info
            self.input_info = sd.query_devices(self.input_device, 'input')
            fs = int(self.input_info['default_samplerate'])
            channels = int(self.input_info['max_input_channels'])
            
            log("AUDIO", f"Starting background recording on {self.input_info['name']} ({fs}Hz, {channels}ch)")
            
            def callback(indata, frames, time_info, status):
                self.in_queue.put(indata[:])

            self.input_stream = sd.RawInputStream(
                samplerate=fs, 
                channels=channels, 
                dtype='int16', 
                device=self.input_device, 
                callback=callback
            )
            self.input_stream.start()
            return True
        except Exception as e:
            log("AUDIO", f"Failed to start background recording: {e}")
            return False

    def stop_background_recording(self):
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

    def record(self, session, max_record_time=10.0, silence_limit=0.8, initial_timeout=3.0):
        log("REC", "Start" if not session.is_speaking else "LISTENING (DURING SPEECH)")
        
        if not self.input_stream or not self.input_stream.active:
            if not self.start_background_recording():
                return False

        # Flush old audio from queue
        while not self.in_queue.empty():
            try: self.in_queue.get_nowait()
            except: break

        recorded_chunks = []
        last_voice_at = time.time()
        voice_detected = False
        start_time = time.time()
        
        fs = int(self.input_info['default_samplerate'])
        channels = int(self.input_info['max_input_channels'])

        try:
            while time.time() - start_time < max_record_time:
                try:
                    # Non-blocking check for new audio chunk (50ms timeout)
                    indata = self.in_queue.get(timeout=0.05)
                    recorded_chunks.append(indata)
                    
                    # Calculate RMS for VAD
                    data = np.frombuffer(indata, dtype=np.int16)
                    rms = np.sqrt(np.mean(data.astype(np.float32)**2))
                    
                    effective_threshold = VAD_THRESHOLD
                    if session.is_speaking:
                        time_since_speech = time.time() - session.speech_start_time
                        if time_since_speech < 2.0:
                            effective_threshold *= 4.0
                        else:
                            effective_threshold *= 2.0
                    
                    if rms > effective_threshold:
                        if session.is_speaking or session.is_generating:
                            log("AUDIO", "INTERRUPTION DETECTED")
                            session.abort()
                        last_voice_at = time.time()
                        voice_detected = True
                    
                    # Check break conditions
                    curr_time = time.time()
                    if voice_detected:
                        if (curr_time - last_voice_at) > silence_limit:
                            break
                    else:
                        if (curr_time - start_time) > initial_timeout:
                            break
                except queue.Empty:
                    continue
        except Exception as e:
            log("AUDIO", f"Recording loop error: {e}")
            return False

        if not recorded_chunks: return False
        audio_bytes = b''.join(recorded_chunks)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
        
        rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
        if rms < SILENCE_THRESHOLD: return False

        write(INPUT_AUDIO, fs, audio)
        log("REC", "End")
        return True

    def is_playing(self):
        """Returns True if there is still audio in the queue or buffer."""
        return self.out_queue.qsize() > 0 or len(self.out_buffer) > 0

    def create_output_stream(self, samplerate=SAMPLERATE):
        # Using a simple buffer to handle Piper's variable chunk sizes
        self.out_buffer = b''
        self.cb_count = 0

        def buffered_callback(outdata, frames, time_info, status):
            needed = len(outdata)
            self.cb_count += 1
            
            while len(self.out_buffer) < needed:
                try:
                    chunk = self.out_queue.get_nowait()
                    self.out_buffer += chunk
                except queue.Empty:
                    break
            
            if len(self.out_buffer) >= needed:
                outdata[:] = self.out_buffer[:needed]
                self.out_buffer = self.out_buffer[needed:]
            else:
                if len(self.out_buffer) > 0:
                    outdata[:len(self.out_buffer)] = self.out_buffer
                    outdata[len(self.out_buffer):] = b'\x00' * (needed - len(self.out_buffer))
                    self.out_buffer = b''
                else:
                    outdata[:] = b'\x00' * needed

        try:
            log("AUDIO", f"Starting callback stream on device {self.output_device}...")
            stream = sd.RawOutputStream(
                samplerate=samplerate,
                channels=1,
                dtype='int16',
                blocksize=2048, # Increased blocksize to prevent micro-stutters during context switching
                device=self.output_device,
                callback=buffered_callback
            )
            stream.start()
            return stream
        except Exception as e:
            log("AUDIO", f"Error creating output stream: {e}")
            raise e
