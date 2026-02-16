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
from queue import Queue
from scipy.io.wavfile import write

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

CPU_THREADS = 6
RECORD_SECONDS = 3
SILENCE_THRESHOLD = 500  # RMS value below which we consider it "silence"

# ============================================
# STATE
# ============================================

speech_queue = Queue()
conversation = []

is_speaking = False
is_generating = False
abort_speech = threading.Event()

SYSTEM_PROMPT = """You are a natural, proactive sales agent for Whatypie. 
Goal: Schedule a demo appointment.
Guidelines:
- Speak like a human on a phone call. Use brief fillers (Um, Well, Got it, I see) sparingly.
- Be human. Use discourse markers (Actually, So, Right, To be fair).
- Be extremely concise but warm. Avoid robotic phrases.
- Do NOT call yourself an AI or chatbot. You are part of the Whatypie team."""

# ============================================
# LOGGER
# ============================================

def log(stage, msg=""):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] [{stage}] {msg}")

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
    while True:
        text = speech_queue.get()
        if text is None: break

        is_speaking = True
        abort_speech.clear()
        
        # Add random "thought" delay for non-filler sentences
        if len(text) > 15:
            time.sleep(random.uniform(0.1, 0.4))

        # Randomize speech speed for human feel
        speed = random.uniform(1.00, 1.10)
        log("TTS", f"({speed:.2f}x) {text}")

        piper = subprocess.Popen(
            [PIPER_PATH, "-m", PIPER_MODEL, "--output_raw", "--length_scale", str(speed)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        piper.stdin.write((text + "\n").encode())
        piper.stdin.close()

        while True:
            if abort_speech.is_set():
                log("SYSTEM", "SPEECH ABORTED")
                piper.terminate()
                break

            chunk = piper.stdout.read(1024)
            if not chunk: break
            audio_stream.write(chunk)

        piper.wait()
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
    
    def callback(indata, frames, time_info, status):
        # Calculate RMS to detect voice
        data = np.frombuffer(indata, dtype=np.int16)
        rms = np.sqrt(np.mean(data.astype(np.float32)**2))
        if rms > 1500: # Threshold for "active" voice
            if is_speaking:
                abort_speech.set()
        recorded_chunks.append(indata[:])

    with sd.RawInputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
        time.sleep(RECORD_SECONDS)

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
    result = subprocess.run(
        [WHISPER_PATH, "-m", WHISPER_MODEL, "-f", INPUT_AUDIO, "-nt", "-t", str(CPU_THREADS)],
        capture_output=True, text=True
    )
    text = re.sub(r'\[.*?\]', '', result.stdout).strip()
    if text: log("USER", text)
    return text

# ============================================
# LLM ASK (CHATS)
# ============================================

def ask_llm(user_text):
    global is_generating
    is_generating = True

    conversation.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation[-10:]

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
                "num_predict": 40,
                "temperature": 0.5,
                "stop": current_stops + ["\nUser", "\nAssistant"]
            }
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

        # Split on sentence boundaries, pauses (,), or semi-colons (;)
        # Threshold lowered to 8 chars for "Instant Reactions" (e.g., "Got it.")
        if len(buffer) > 8:
            sentences = re.findall(r'[^.!?]*[.!?]|[^,;]*[,;]', buffer)
        else:
            sentences = re.findall(r'[^.!?]*[.!?]', buffer)

        for s in sentences:
            clean = s.strip()
            if len(clean) > 2:
                log("AGENT", clean)
                speech_queue.put(clean)
                full_response.append(clean)

        buffer = re.sub(r'[^.!?]*[.!?]|[^,;]*[,;]' if len(buffer) > 8 else r'[^.!?]*[.!?]', '', buffer)

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
                ask_llm(text)

# ============================================

if __name__ == "__main__":
    run()
