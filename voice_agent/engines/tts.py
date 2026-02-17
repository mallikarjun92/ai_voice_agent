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
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0
        )
        
        def reader():
            while self.process and self.process.poll() is None:
                try:
                    chunk = self.process.stdout.read(4096)
                    if not chunk: break
                    self.audio_queue.put(chunk)
                except: break
        threading.Thread(target=reader, daemon=True).start()

    def speak(self, text):
        if not self.process or self.process.poll() is not None:
            self.start(self.last_speed)
        try:
            self.process.stdin.write((text.strip() + "\n").encode())
            self.process.stdin.flush()
        except:
            self.start(self.last_speed)
            self.process.stdin.write((text.strip() + "\n").encode())
            self.process.stdin.flush()

    def cleanup(self):
        if self.process:
            self.process.terminate()

class TTSEngine:
    def __init__(self, audio_stream, session):
        self.audio_stream = audio_stream
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
                    chunk = self.engine.audio_queue.get(timeout=0.01)
                    try:
                        self.audio_stream.write(chunk)
                    except sd.PortAudioError:
                        log("SYSTEM", "Restarting audio stream...")
                        try: self.audio_stream.stop(); self.audio_stream.start()
                        except: pass
                        self.audio_stream.write(chunk)
                    
                    last_audio_at = time.time()
                    self.session.is_speaking = True
                except queue.Empty:
                    if not self.speech_queue.empty():
                        try:
                            next_part = self.speech_queue.get_nowait()
                            if next_part:
                                log("TTS", next_part)
                                self.engine.speak(next_part)
                            self.speech_queue.task_done()
                        except: pass
                        continue

                    if time.time() - last_audio_at > 0.4:
                        break
                    if not self.session.is_generating and (time.time() - last_audio_at > 1.5):
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
        try:
            self.audio_stream.stop()
            self.audio_stream.abort()
            self.audio_stream.start()
        except: pass
