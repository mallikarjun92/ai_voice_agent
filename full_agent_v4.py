import sounddevice as sd
import subprocess
import requests
import threading
import time
import re
import json
import random
import numpy as np
from datetime import datetime
import queue
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

# ============================================
# CONFIG
# ============================================

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-tiny.en.bin"

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

INPUT_AUDIO = r"C:\VOICE_AGENT\input.wav"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:0.5b"

STOP_TOKENS_MAP = {
    "gemma": ["<start_of_turn>", "<end_of_turn>", "user", "model", "User:", "Assistant:"],
    "qwen": ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "User:", "Assistant:"]
}

CPU_THREADS = 4
RECORD_SECONDS = 3
SILENCE_THRESHOLD = 500  # RMS value below which we consider it "silence"

# ============================================
# STATE
# ============================================

speech_queue = queue.Queue()
conversation = []

is_speaking = False
is_generating = False
abort_speech = threading.Event()

SYSTEM_PROMPT = "You are a Whatypie sales agent. Be human, concise, and warm. Goal: Schedule a demo."

INSTANT_FILLERS = ["Got it.", "I see.", "Okay.", "Right.", "Mm-hmm.", "Uh-huh.", "Gotcha."]

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
            try: self.process.terminate()
            except: pass
        self.last_speed = speed
        self.process = subprocess.Popen(
            [self.path, "-m", self.model, "--output_raw", "--length_scale", str(speed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0 # Unbuffered for speed
        )
        # Background thread to read stdout without blocking main TTS thread
        def reader():
            while self.process and self.process.poll() is None:
                chunk = self.process.stdout.read(4096)
                if not chunk: break
                self.audio_queue.put(chunk)
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

def sanitize_text(text):
    """Removes emojis, markdown, and special characters for clean TTS."""
    # Remove Emojis and non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove Markdown symbols like *, _, #
    text = re.sub(r'[*_#`~\[\]()]', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================
# LOGGER
# ============================================

def log(stage, msg=""):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] [{stage}] {msg}")

# Pre-load models for zero-latency startup
log("STT", "Loading Faster-Whisper...")
stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
log("STT", "Ready")

# ============================================
# AUDIO STREAM (ONE ONLY)
# ============================================

audio_stream = sd.RawOutputStream(
    samplerate=22050,
    channels=1,
    dtype='int16',
    blocksize=4096
)
audio_stream.start()
log("AUDIO", "Ready")

# ============================================
# TTS WORKER
# ============================================

def tts_worker():
    global is_speaking
    engine = PiperEngine(PIPER_PATH, PIPER_MODEL)
    
    while True:
        try:
            text = speech_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        if text is None: break

        # Abortion Guard
        if abort_speech.is_set():
            while not speech_queue.empty():
                try: speech_queue.get_nowait(); speech_queue.task_done()
                except: break
            while not engine.audio_queue.empty(): engine.audio_queue.get_nowait()
            abort_speech.clear()
            continue

        is_speaking = True
        
        # Batch any immediate words
        items = [text]
        while not speech_queue.empty():
            try:
                extra = speech_queue.get_nowait()
                if extra: items.append(extra)
                speech_queue.task_done()
            except: break
        
        combined_text = " ".join(items)
        log("TTS", f"{combined_text}")
        engine.speak(combined_text)

        # Audio read loop
        last_audio_at = time.time()
        while True:
            if abort_speech.is_set():
                log("SYSTEM", "SPEECH ABORTED")
                engine.start(engine.last_speed) 
                while not speech_queue.empty():
                    try: speech_queue.get_nowait(); speech_queue.task_done()
                    except: break
                break
            
            try:
                chunk = engine.audio_queue.get(timeout=0.05)
                audio_stream.write(chunk)
                last_audio_at = time.time()
            except queue.Empty:
                # CRITICAL: If audio is empty but more TEXT is in the queue,
                # break immediately to send the new text to the engine.
                if not speech_queue.empty():
                    break
                    
                # If LLM is done and we haven't heard audio for 300ms, assume sentence done.
                if not is_generating and (time.time() - last_audio_at > 0.3):
                    break
                
                # Check for extremely slow generation
                if time.time() - last_audio_at > 2.0:
                    break
                continue

        is_speaking = False
        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# ============================================
# RECORD & INTERRUPT DETECTION
# ============================================

def record():
    global is_speaking
    
    log("REC", "Start" if not is_speaking else "LISTENING (DURING SPEECH)")
    fs = 16000
    recorded_chunks = []
    
    # VAD State
    last_voice_at = time.time()
    voice_detected = False
    max_record_time = 10.0 # Safety limit
    silence_limit = 0.8    # Stop after 800ms of silence
    initial_timeout = 3.0  # Stop if no voice after 3s
    
    def callback(indata, frames, time_info, status):
        nonlocal last_voice_at, voice_detected
        # Calculate RMS to detect voice
        data = np.frombuffer(indata, dtype=np.int16)
        rms = np.sqrt(np.mean(data.astype(np.float32)**2))
        
        if rms > 1500: # Threshold for "active" voice
            if is_speaking:
                abort_speech.set()
            last_voice_at = time.time()
            voice_detected = True
            
        recorded_chunks.append(indata[:])

    with sd.RawInputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
        start_time = time.time()
        while time.time() - start_time < max_record_time:
            time.sleep(0.05)
            curr_time = time.time()
            
            # Scenario A: We heard something, now wait for silence
            if voice_detected:
                if (curr_time - last_voice_at) > silence_limit:
                    break
            # Scenario B: No voice detected at all for a while
            else:
                if (curr_time - start_time) > initial_timeout:
                    break

    if not recorded_chunks: return False
    audio = np.frombuffer(b''.join(recorded_chunks), dtype=np.int16)
    
    rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
    if rms < SILENCE_THRESHOLD: return False

    write(INPUT_AUDIO, fs, audio)
    log("REC", "End")
    return True

# ============================================
# TRANSCRIBE
# ============================================

def transcribe():
    segments, info = stt_model.transcribe(INPUT_AUDIO, beam_size=1)
    text = " ".join([segment.text for segment in segments]).strip()
    if text: log("USER", text)
    return text

# ============================================
# LLM ASK (CHATS)
# ============================================

def ask_llm(user_text):
    global is_generating
    is_generating = True

    conversation.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation[-4:]

    # Get tokens for current model
    current_stops = []
    for key, tokens in STOP_TOKENS_MAP.items():
        if key in MODEL_NAME.lower():
            current_stops = tokens
            break

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,
            "options": {
                "num_thread": CPU_THREADS,
                "num_predict": 50,
                "temperature": 0.4,
                "num_ctx": 2048,
                "stop": current_stops + ["\nUser", "\nAssistant"]
            },
            "keep_alive": "5m"
        },
        stream=True
    )

    buffer = ""
    full_response = []

    for line in response.iter_lines():
        if not line: continue
        token = json.loads(line.decode())["message"]["content"]
        
        # Dynamic Hallucination Guard
        lower_token = token.lower()
        active_guards = [s.lower() for s in current_stops if len(s) > 2]
        if any(x in lower_token for x in active_guards):
            log("SYSTEM", f"Hallucination Guard Triggered: {token}")
            break

        buffer += token

        # Sentence-Based Streaming: Buffer until punctuation for natural intonation
        if any(p in buffer for p in ".!?"):
            # Split at the last punctuation mark
            match = re.search(r'(.*[.!?])', buffer)
            if match:
                sentence = match.group(0).strip()
                if sentence:
                    clean_sent = sanitize_text(sentence)
                    if clean_sent:
                        log("AGENT", f"(Stream) {clean_sent}")
                        speech_queue.put(clean_sent)
                        full_response.append(clean_sent)
                
                buffer = buffer[len(match.group(0)):]
                continue

    # Final flush for the last bit of the response
    if buffer.strip():
        clean_last = sanitize_text(buffer.strip())
        if clean_last:
            log("AGENT", f"(Final) {clean_last}")
            speech_queue.put(clean_last)
            full_response.append(clean_last)

    if full_response:
        conversation.append({"role": "assistant", "content": " ".join(full_response)})
    
    is_generating = False

# ============================================
# PRELOAD
# ============================================

def preload():
    log("LLM", "Loading")
    requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "messages": [], "keep_alive": "24h"})
    log("LLM", "Ready")

# ============================================
# MAIN LOOP
# ============================================

def run():
    preload()
    log("SYSTEM", "READY")
    # Announce system status to user
    speech_queue.put("System is ready.")
    
    while True:
        has_sound = record()
        if has_sound:
            text = transcribe()
            if text:
                if is_speaking: abort_speech.set()
                
                # Immediate acknowledgment (Instant Filler)
                filler = random.choice(INSTANT_FILLERS)
                log("AGENT", f"(Instant Filler) {filler}")
                speech_queue.put(filler)
                
                ask_llm(text)

# ============================================

if __name__ == "__main__":
    run()
