import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import requests
import time
import re
import json
import threading
import queue
from datetime import datetime

# ============================================
# CONFIG
# ============================================

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-tiny.en.bin"

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

INPUT_AUDIO = r"C:\VOICE_AGENT\input.wav"
OUTPUT_AUDIO = r"C:\VOICE_AGENT\output.wav"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

CPU_THREADS = 10
RECORD_SECONDS = 2.5

conversation_history = []
call_started = False

speech_queue = queue.Queue()

# ============================================
# LOGGER
# ============================================

def log(msg):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] {msg}")

# ============================================
# CLEAN RESPONSE
# ============================================

def clean_response(text):

    text = re.sub(r'^Agent:\s*', '', text.strip())

    sentences = re.split(r'(?<=[.!?])\s+', text)

    if sentences:
        return sentences[0]

    return text

# ============================================
# PIPER PERSISTENT WORKER (NO RELOAD)
# ============================================

def speech_worker():

    while True:

        text = speech_queue.get()

        if text is None:
            break

        log(f"TTS speaking: {text}")

        # Generate audio
        subprocess.run(
            f'echo {text} | "{PIPER_PATH}" -m "{PIPER_MODEL}" -f "{OUTPUT_AUDIO}"',
            shell=True
        )

        # Play audio
        subprocess.run(
            f'powershell -c (New-Object Media.SoundPlayer "{OUTPUT_AUDIO}").PlaySync();',
            shell=True
        )

        speech_queue.task_done()

def speak(text):

    speech_queue.put(text)

# ============================================
# RECORD
# ============================================

def record_audio():

    log("Recording started")

    fs = 16000

    recording = sd.rec(
        int(RECORD_SECONDS * fs),
        samplerate=fs,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    write(INPUT_AUDIO, fs, recording)

    log("Recording finished")

# ============================================
# TRANSCRIBE
# ============================================

def transcribe():

    start = time.time()

    result = subprocess.run(
        f'"{WHISPER_PATH}" -m "{WHISPER_MODEL}" -f "{INPUT_AUDIO}" -nt -t {CPU_THREADS}',
        shell=True,
        capture_output=True,
        text=True
    )

    text = re.sub(r'\[.*?\]', '', result.stdout).strip()

    log(f"Transcription done ({time.time()-start:.2f}s)")

    return text

# ============================================
# STREAM LLM WITH EARLY SPEECH
# ============================================

def ask_llm_stream(user_text):

    global call_started, conversation_history

    conversation_history.append(f"Customer: {user_text}")
    conversation_history = conversation_history[-2:]

    if not call_started:
        instruction = "Greet briefly and introduce yourself in under 8 words."
        call_started = True
    else:
        instruction = "Reply briefly in under 12 words."

    prompt = (
        "You are a cold calling sales agent.\n"
        "Speak short, natural sentences.\n\n"
        + instruction + "\n\n"
        + "\n".join(conversation_history)
        + "\nAgent:"
    )

    log("LLM streaming started")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "num_predict": 30,
                "temperature": 0.3,
                "num_thread": CPU_THREADS
            }
        },
        stream=True
    )

    full_text = ""
    first_spoken = False
    start_time = time.time()

    for line in response.iter_lines():

        if not line:
            continue

        data = json.loads(line)
        token = data.get("response", "")

        full_text += token

        if not first_spoken:

            sentences = re.split(r'(?<=[.!?])\s+', full_text)

            if sentences and sentences[0].endswith((".", "!", "?")):

                sentence = clean_response(sentences[0])

                log(f"Speaking early at {time.time()-start_time:.2f}s")

                speak(sentence)

                first_spoken = True

    reply = clean_response(full_text)

    conversation_history.append(f"Agent: {reply}")

    log(f"LLM finished ({time.time()-start_time:.2f}s)")

    return reply

# ============================================
# MAIN LOOP
# ============================================

def run_agent():

    print("\nULTRA-LOW-LATENCY VOICE AGENT STARTED\n")

    while True:

        try:

            record_audio()

            user_text = transcribe()

            if not user_text:
                continue

            print("\nCustomer:", user_text)

            reply = ask_llm_stream(user_text)

            print("Agent:", reply)

            print("\n--- Conversation ---")
            for line in conversation_history:
                print(line)
            print("--------------------\n")

        except KeyboardInterrupt:

            print("\nStopping...")

            speech_queue.put(None)

            break

# ============================================

threading.Thread(target=speech_worker, daemon=True).start()

if __name__ == "__main__":
    run_agent()
