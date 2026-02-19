import sounddevice as sd
import subprocess
import requests
import threading
import time
import re
import json
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

SYSTEM_PROMPT = """You are a natural, proactive sales agent for Whatypie. 
Goal: Schedule a demo appointment.
Guidelines:
- Speak like a human on a phone call. Use brief fillers (Um, Well, Got it, I see) sparingly.
- Be extremely concise but warm. Avoid robotic phrases.
- Do NOT call yourself an AI or chatbot. You are part of the Whatypie team."""

is_speaking = False
is_generating = False

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

        if text is None:
            break

        is_speaking = True

        log("TTS", text)

        piper = subprocess.Popen(
            [
                PIPER_PATH,
                "-m",
                PIPER_MODEL,
                "--output_raw"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        piper.stdin.write((text + "\n").encode())
        piper.stdin.close()

        while True:

            chunk = piper.stdout.read(4096)

            if not chunk:
                break

            audio_stream.write(chunk)

        piper.wait()

        is_speaking = False

        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# ============================================
# RECORD
# ============================================

def record():

    global is_speaking

    # DO NOT RECORD while speaking
    while is_speaking:
        time.sleep(0.05)

    log("REC", "Start")

    fs = 16000

    audio = sd.rec(
        int(RECORD_SECONDS * fs),
        samplerate=fs,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    write(INPUT_AUDIO, fs, audio)

    log("REC", "End")

    # Basic VAD: Return True if there is actual sound, False if it's mostly silence
    rms = np.sqrt(np.mean(audio.astype(np.int32)**2))
    if rms < SILENCE_THRESHOLD:
        return False
    
    return True

# ============================================
# TRANSCRIBE
# ============================================

def transcribe():

    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m",
            WHISPER_MODEL,
            "-f",
            INPUT_AUDIO,
            "-nt",
            "-t",
            str(CPU_THREADS)
        ],
        capture_output=True,
        text=True
    )

    text = re.sub(r'\[.*?\]', '', result.stdout).strip()

    if text:
        log("USER", text)

    return text

# ============================================
# LLM STREAM WITH MEMORY
# ============================================

def ask_llm(user_text):

    global is_generating

    is_generating = True

    conversation.append({"role": "user", "content": user_text})
    
    # Keep last 10 messages
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

        if not line:
            continue

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

    # After full response is streamed, add it as ONE entry to history
    if full_response:
        combined = ' '.join(full_response)
        # Final cleanup in case labels leaked through
        combined = re.sub(r'^(User|Assistant):\s*', '', combined)
        conversation.append({"role": "assistant", "content": combined})

    is_generating = False

# ============================================
# PRELOAD MODEL
# ============================================

def preload():

    log("LLM", "Loading")

    requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hi"}],
            "keep_alive": "24h",
            "options": {"num_thread": CPU_THREADS}
        }
    )

    log("LLM", "Ready")

# ============================================
# MAIN LOOP
# ============================================

def run():

    preload()

    log("SYSTEM", "READY")

    while True:

        if is_generating or is_speaking:
            time.sleep(0.05)
            continue

        has_sound = record()

        if not has_sound:
            continue

        text = transcribe()

        if text:
            ask_llm(text)

# ============================================

run()
