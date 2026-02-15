import sounddevice as sd
import subprocess
import requests
import threading
import time
import re
import json
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

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"

CPU_THREADS = 10
RECORD_SECONDS = 2

# ============================================
# STATE
# ============================================

speech_queue = Queue()

conversation = []

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

    conversation.append(f"Customer: {user_text}")
    conversation[:] = conversation[-6:]

    prompt = f"""
You are a WhatsApp automation sales agent.

Speak naturally.

Conversation:
{chr(10).join(conversation)}

Agent:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_thread": CPU_THREADS,
                "num_predict": 30,
                "temperature": 0.4
            }
        },
        stream=True
    )

    buffer = ""

    for line in response.iter_lines():

        if not line:
            continue

        token = json.loads(line.decode())["response"]

        buffer += token

        sentences = re.findall(r'[^.!?]*[.!?]', buffer)

        for s in sentences:

            clean = s.strip()

            if len(clean) > 3:

                log("AGENT", clean)

                speech_queue.put(clean)

                conversation.append(f"Agent: {clean}")

        buffer = re.sub(r'[^.!?]*[.!?]', '', buffer)

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
            "prompt": "hi",
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

        record()

        text = transcribe()

        if text:
            ask_llm(text)

# ============================================

run()
