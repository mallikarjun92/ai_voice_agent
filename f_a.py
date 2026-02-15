import sounddevice as sd
import numpy as np
import subprocess
import requests
import json
import threading
import queue
import re
import time
import sys
from collections import deque

# ============================================
# CONFIGURATION (tuned for your hardware)
# ============================================

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-tiny.en.bin"

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
ENERGY_THRESHOLD = 0.02          # adjust based on your mic sensitivity
SILENCE_DURATION = 1.0            # seconds of silence to end utterance

NUM_WHISPER_THREADS = 10
NUM_LLM_THREADS = 10

# ============================================
# SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are a human sales agent making a cold call to offer WhatsApp automation software.

The call is LIVE.

Your behavior rules:

FIRST MESSAGE:
- Greet properly
- Introduce yourself
- Say why you are calling

NEXT MESSAGES:
- NEVER greet again
- Continue conversation naturally
- Answer questions directly
- Ask follow-up questions
- If unclear input, ask clarification

GOAL:
Book a demo.

Speak naturally like human.
Max 2 sentences.
"""

# ============================================
# GLOBAL STATE & QUEUES
# ============================================

conversation_history = deque(maxlen=8)
call_started = False
stop_event = threading.Event()

audio_queue = queue.Queue()          # raw audio chunks
text_queue = queue.Queue()           # transcribed user text
response_queue = queue.Queue()       # LLM response sentences
interrupt_playback = threading.Event()

# ============================================
# VAD: RECORD WITH SILENCE DETECTION (FIXED)
# ============================================

def record_until_silence():
    audio_buffer = bytearray()
    silent_chunks = 0
    max_silent_chunks = int(SILENCE_DURATION * SAMPLE_RATE / BLOCK_SIZE)
    recording = False

    def callback(indata, frames, time_info, status):
        nonlocal audio_buffer, silent_chunks, recording
        if status:
            print(f"Audio error: {status}", file=sys.stderr)
            return

        # Convert to float32 for safe energy calculation (fixes overflow warning)
        audio_float = indata.astype(np.float32)
        energy = np.sqrt(np.mean(audio_float**2))

        if energy > ENERGY_THRESHOLD:
            silent_chunks = 0
            if not recording:
                recording = True
                print("Speech started...")
            audio_buffer.extend(indata.tobytes())
        else:
            if recording:
                silent_chunks += 1
                audio_buffer.extend(indata.tobytes())   # keep a bit of silence for context
                if silent_chunks > max_silent_chunks:
                    # End of utterance
                    audio_queue.put(bytes(audio_buffer))
                    audio_buffer.clear()
                    recording = False
                    silent_chunks = 0
                    print("Silence detected, sending for transcription.")
            # else ignore silence

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        blocksize=BLOCK_SIZE, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)

# ============================================
# TRANSCRIPTION WORKER
# ============================================

def transcribe_worker():
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        temp_file = f"temp_audio_{time.time_ns()}.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_data)

        cmd = f'"{WHISPER_PATH}" -m "{WHISPER_MODEL}" -f "{temp_file}" -nt --threads {NUM_WHISPER_THREADS}'
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # Clean up temp file
        subprocess.run(f'del {temp_file}', shell=True, capture_output=True)

        raw = proc.stdout.strip()
        if not raw:
            continue

        text = re.sub(r'\[.*?\]', '', raw).strip()
        if text:
            text_queue.put(text)
            print(f"\nCustomer: {text}")

# ============================================
# LLM WORKER (STREAMING)
# ============================================

def is_unclear(text):
    if len(text) < 3:
        return True
    unclear_words = ["okay", "ok", "hmm", "uh", "ah", "yes", "no", "peace"]
    if text.lower().strip() in unclear_words:
        return True
    return False

def ask_llm_streaming(user_text):
    global conversation_history, call_started

    if is_unclear(user_text):
        yield "Sorry, could you please clarify that?"
        return

    conversation_history.append(f"Customer: {user_text}")

    if not call_started:
        instruction = "This is the FIRST message. Greet and introduce yourself."
        call_started = True
    else:
        instruction = "Continue conversation naturally without greeting."

    prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + instruction
        + "\n\n"
        + "\n".join(conversation_history)
        + "\nAgent:"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.6,
            "num_predict": 80
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        buffer = ""
        sentence_end = re.compile(r'[.!?]')

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'response' in data:
                    token = data['response']
                    buffer += token

                    if sentence_end.search(token) and len(buffer.strip()) > 10:
                        yield buffer.strip()
                        buffer = ""

        if buffer.strip():
            yield buffer.strip()

    except Exception as e:
        print(f"LLM error: {e}")
        yield "I'm sorry, I'm having trouble responding right now."

def llm_worker():
    while not stop_event.is_set():
        try:
            user_text = text_queue.get(timeout=1)
        except queue.Empty:
            continue

        for sentence in ask_llm_streaming(user_text):
            response_queue.put(sentence)
            conversation_history.append(f"Agent: {sentence}")

# ============================================
# TTS WORKER (with interrupt support)
# ============================================

def play_audio(file_path):
    process = subprocess.Popen(
        f'powershell -c (New-Object Media.SoundPlayer "{file_path}").PlaySync();',
        shell=True
    )
    while process.poll() is None:
        if interrupt_playback.is_set():
            process.terminate()
            break
        time.sleep(0.05)

def tts_worker():
    while not stop_event.is_set():
        try:
            text = response_queue.get(timeout=1)
        except queue.Empty:
            continue

        if interrupt_playback.is_set():
            interrupt_playback.clear()
            continue

        output_file = f"temp_tts_{time.time_ns()}.wav"
        cmd = f'echo "{text}" | "{PIPER_PATH}" -m "{PIPER_MODEL}" -f "{output_file}"'
        subprocess.run(cmd, shell=True, capture_output=True)

        play_thread = threading.Thread(target=play_audio, args=(output_file,))
        play_thread.start()
        play_thread.join()

        subprocess.run(f'del {output_file}', shell=True, capture_output=True)

# ============================================
# BARGE-IN DETECTION
# ============================================

def barge_in_listener():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        blocksize=BLOCK_SIZE) as stream:
        while not stop_event.is_set():
            data, overflowed = stream.read(BLOCK_SIZE)
            # Convert to float32 for safe energy calculation
            audio_float = data.astype(np.float32)
            energy = np.sqrt(np.mean(audio_float**2))
            if energy > ENERGY_THRESHOLD:
                interrupt_playback.set()
                time.sleep(0.2)   # debounce

# ============================================
# MAIN
# ============================================

def run_agent():
    print("\n🚀 AI VOICE AGENT STARTED (streaming, interruptible)")
    print("Press Ctrl+C to stop.\n")

    threads = [
        threading.Thread(target=record_until_silence, name="Recorder"),
        threading.Thread(target=transcribe_worker, name="Transcriber"),
        threading.Thread(target=llm_worker, name="LLM"),
        threading.Thread(target=tts_worker, name="TTS"),
        threading.Thread(target=barge_in_listener, name="BargeIn")
    ]

    for t in threads:
        t.daemon = True
        t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        for t in threads:
            t.join(timeout=2)
        print("Agent stopped.")

if __name__ == "__main__":
    run_agent()