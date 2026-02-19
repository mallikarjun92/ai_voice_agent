import subprocess
import threading
import queue
import time
import sounddevice as sd
from voice_agent.utils.logger import log
from voice_agent.config import PIPER_PATH, PIPER_MODEL, SAMPLERATE

class PiperEngine:
    def __init__(self, path, model):
        self.path = path
        self.model = model
        self.process = None
        self.last_speed = 1.0
        self.audio_queue = queue.Queue()
        self.start()

    def start(self, speed=1.0):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=0.2)
            except:
                pass
        
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except: break

        self.last_speed = speed
        self.process = subprocess.Popen(
            [self.path, "-m", self.model, "--output_raw", "--length_scale", str(speed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0
        )
        
        def reader():
            try:
                while self.process and self.process.poll() is None:
                    # Reverting to read() as read1() is not available on FileIO
                    # Using a smaller chunk size to reduce latency
                    chunk = self.process.stdout.read(1024)
                    if not chunk: 
                        log("PIPER", "End of stdout stream (Piper closed)")
                        break
                    self.audio_queue.put(chunk)
            except Exception as e:
                log("PIPER", f"Reader thread error: {e}")
        threading.Thread(target=reader, daemon=True).start()

        def err_reader():
            try:
                while self.process and self.process.poll() is None:
                    line = self.process.stderr.readline()
                    if not line: break
                    log("PIPER_ERR", line.decode().strip())
            except: pass
        threading.Thread(target=err_reader, daemon=True).start()

    def speak(self, text):
        if not self.process or self.process.poll() is not None:
            log("PIPER", "Restarting process for new text.")
            self.start(self.last_speed)
        try:
            log("PIPER", f"Speaking: [{text.strip()}]")
            self.process.stdin.write((text.strip() + "\n").encode())
            self.process.stdin.flush()
        except Exception as e:
            log("PIPER", f"Write error: {e}")
            self.start(self.last_speed)
            self.process.stdin.write((text.strip() + "\n").encode())
            self.process.stdin.flush()

    def cleanup(self):
        if self.process:
            self.process.terminate()

class TTSEngine:
    def __init__(self, audio_handler, session):
        self.audio_handler = audio_handler
        self.session = session
        self.speech_queue = queue.Queue()
        self.engine = PiperEngine(PIPER_PATH, PIPER_MODEL)
        
        # Start worker thread
        threading.Thread(target=self._worker, daemon=True).start()

    def put(self, text):
        self.speech_queue.put(text)

    def _worker(self):
        last_audio_at = time.time()
        
        while True:
            if time.time() - last_audio_at > 0.3:
                # Only lower the flag if the audio handler itself is empty
                if not self.audio_handler.is_playing():
                    self.session.is_speaking = False

            try:
                text = self.speech_queue.get(timeout=0.05)
            except queue.Empty:
                continue
                
            if text is None: break

            if self.session.abort_speech.is_set():
                self._purge_queues()
                self.session.abort_speech.clear()
                continue

            self.session.is_speaking = True
            self.session.speech_start_time = time.time()
            log("TTS", f"{text}")
            self.engine.speak(text)

            last_audio_at = time.time()
            while True:
                if self.session.abort_speech.is_set():
                    log("SYSTEM", "SPEECH ABORTED")
                    self._purge_queues()
                    # Re-start engine to clear piper buffer
                    self.engine.start(self.engine.last_speed)
                    break
                
                try:
                    chunk = self.engine.audio_queue.get(timeout=0.05)
                    # Feed the audio handler's queue for callback-based playback
                    self.audio_handler.out_queue.put(chunk)
                    
                    last_audio_at = time.time()
                    self.session.is_speaking = True
                except queue.Empty:
                    # Check if there's more text coming in the speech queue
                    if not self.speech_queue.empty():
                        try:
                            next_part = self.speech_queue.get_nowait()
                            if next_part:
                                log("TTS", next_part)
                                self.engine.speak(next_part)
                                last_audio_at = time.time()
                            self.speech_queue.task_done()
                        except: pass
                        continue

                    # Timeout logic:
                    # 1. If we are still generating text, be very patient (waiting for LLM)
                    # 2. If we are done generating, wait up to 2 seconds for Piper to finish its buffer
                    elapsed = time.time() - last_audio_at
                    
                    if self.session.is_generating:
                        if elapsed > 5.0: # Patient with LLM (Max wait)
                            log("TTS_DEBUG", "Worker timed out waiting for LLM audio.")
                            break
                    else:
                        if elapsed > 2.0: # Patient with Piper (Max wait)
                            log("TTS_DEBUG", "Worker timed out waiting for Piper finish.")
                            break
                    continue

            self.speech_queue.task_done()

    def _purge_queues(self):
        while not self.speech_queue.empty():
            try: self.speech_queue.get_nowait(); self.speech_queue.task_done()
            except: break
        while not self.engine.audio_queue.empty():
            try: self.engine.audio_queue.get_nowait()
            except: break
        while not self.audio_handler.out_queue.empty():
            try: self.audio_handler.out_queue.get_nowait()
            except: break
        # The stream itself is now managed by AudioHandler, but we can stop/start it if needed via session logic if we had it.
        # However, clearing the queue is usually enough for the callback to start playing silence.
