import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import requests
import threading
import time
import re
import json
from datetime import datetime
from queue import Queue

# ============================================
# CONFIG
# ============================================

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-tiny.en.bin"

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

INPUT_AUDIO = "input.wav"
OUTPUT_AUDIO = "output.wav"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

CPU_THREADS = 10
RECORD_SECONDS = 2.0

# ============================================
# STATE
# ============================================

conversation_history = []
call_started = False
speech_queue = Queue()

agent_busy = False

# ============================================
# LOGGER
# ============================================

def log(stage, msg=""):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] [{stage}] {msg}")

# ============================================
# SPEECH WORKER
# ============================================

def speech_worker():

    while True:

        text = speech_queue.get()

        if text is None:
            break

        log("TTS", f"Generating: {text}")

        subprocess.run(
            f'echo {text} | "{PIPER_PATH}" -m "{PIPER_MODEL}" -f "{OUTPUT_AUDIO}"',
            shell=True
        )

        log("TTS", "Generation complete")

        log("PLAYBACK", "Starting")

        subprocess.run(
            f'powershell -c (New-Object Media.SoundPlayer "{OUTPUT_AUDIO}").PlaySync();',
            shell=True
        )

        log("PLAYBACK", "Finished")

        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

# ============================================
# RECORD
# ============================================

def record_audio():

    log("RECORD", "Start")

    fs = 16000

    recording = sd.rec(
        int(RECORD_SECONDS * fs),
        samplerate=fs,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    write(INPUT_AUDIO, fs, recording)

    log("RECORD", "End")

# ============================================
# TRANSCRIBE
# ============================================

def transcribe():

    log("STT", "Start")

    start = time.time()

    result = subprocess.run(
        f'"{WHISPER_PATH}" -m "{WHISPER_MODEL}" -f "{INPUT_AUDIO}" -nt -t {CPU_THREADS}',
        shell=True,
        capture_output=True,
        text=True
    )

    text = re.sub(r'\[.*?\]', '', result.stdout).strip()

    log("STT", f"End ({time.time()-start:.2f}s): {text}")

    return text

# ============================================
# LLM STREAM
# ============================================

def ask_llm_stream(user_text):

    global conversation_history, call_started, agent_busy

    agent_busy = True

    log("LLM", "Start streaming")

    conversation_history.append(f"Customer: {user_text}")
    conversation_history = conversation_history[-6:]

    if not call_started:
        instruction = "Greet and introduce yourself."
        call_started = True
    else:
        instruction = "Continue naturally."

    prompt = (
        "You are human sales agent.\n"
        "Max 2 sentences.\n\n"
        + instruction
        + "\n\n"
        + "\n".join(conversation_history)
        + "\nAgent:"
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 60,
                "temperature": 0.6,
                "num_thread": CPU_THREADS
            }
        },
        stream=True
    )

    full = ""
    spoken = False

    for line in response.iter_lines():

        if not line:
            continue

        data = json.loads(line)

        token = data.get("response", "")

        if token:
            log("LLM", f"TOKEN: {token}")

        full += token

        sentences = re.split(r'(?<=[.!?])\s+', full)

        if sentences and not spoken and len(sentences[0]) > 10:

            log("LLM", f"SPEAK TRIGGER: {sentences[0]}")

            speech_queue.put(sentences[0])

            spoken = True

    log("LLM", "Finished")

    conversation_history.append(f"Agent: {full}")

    agent_busy = False

# ============================================
# MAIN LOOP
# ============================================

def run_agent():

    log("SYSTEM", "VOICE AGENT STARTED")

    while True:

        if agent_busy:

            time.sleep(0.1)
            continue

        record_audio()

        text = transcribe()

        if not text:
            continue

        log("USER", text)

        ask_llm_stream(text)

# ============================================

run_agent()
