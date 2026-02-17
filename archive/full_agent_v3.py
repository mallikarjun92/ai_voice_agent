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
import os
import winsound
import numpy as np
import queue

# ============================================
# CONFIG
# ============================================

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-tiny.en.bin"

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

OUTPUT_AUDIO = r"C:\VOICE_AGENT\piper\output.wav"
INPUT_AUDIO = r"C:\VOICE_AGENT\input.wav"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"

CPU_THREADS = 10
RECORD_SECONDS = 2

# ============================================
# STATE
# ============================================

conversation_history = []
# speech_queue = Queue()
speech_queue = Queue()

tts_lock = threading.Lock()

is_speaking = False

audio_queue = queue.Queue()

is_speaking = False
is_generating = False

# piper_process = None

# ============================================
# LOGGER
# ============================================

def log(stage, msg=""):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}] [{stage}] {msg}")

# ============================================
# PRELOAD MODEL
# ============================================

def start_ollama():

    log("LLM", "Preloading model...")

    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "hi",
                "keep_alive": "24h",
                "options": {
                    "num_thread": CPU_THREADS,
                    "num_predict": 5
                }
            },
            timeout=10
        )

        log("LLM", "Model ready")

    except Exception as e:
        log("LLM ERROR", str(e))

# ============================================
# START PIPER
# ============================================

def start_piper():

    global piper_process

    log("TTS", "Starting Piper...")

    piper_process = subprocess.Popen(
        [
            PIPER_PATH,
            "-m", PIPER_MODEL,
            "-f", OUTPUT_AUDIO
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    time.sleep(0.5)

    log("TTS", "Piper ready")

# ============================================
# SPEAK
# ============================================

def speak(text):

    global is_speaking

    try:

        is_speaking = True

        log("TTS", f"Streaming: {text}")

        start = time.time()

        # Start Piper streaming raw PCM
        piper = subprocess.Popen(
            [
                PIPER_PATH,
                "-m", PIPER_MODEL,
                "--output_raw"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        # Send text
        piper.stdin.write((text + "\n").encode())
        piper.stdin.close()

        samplerate = 22050
        channels = 1
        dtype = 'int16'

        # Open continuous output stream
        with sd.RawOutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            blocksize=4096
        ) as stream:

            log("PLAYBACK", "Streaming to speaker")

            while True:

                chunk = piper.stdout.read(4096)

                if not chunk:
                    break

                stream.write(chunk)

        log("PLAYBACK", f"Done ({time.time()-start:.2f}s)")

    except Exception as e:

        log("TTS ERROR", str(e))

    finally:

        is_speaking = False


# ============================================
# TTS THREAD
# ============================================

def tts_worker():

    global is_speaking

    samplerate = 22050

    stream = sd.RawOutputStream(
        samplerate=samplerate,
        channels=1,
        dtype='int16',
        blocksize=4096
    )

    stream.start()

    log("TTS", "Audio stream started")

    while True:

        text = speech_queue.get()

        if text is None:
            break

        try:

            is_speaking = True

            log("TTS", f"Synthesizing: {text}")

            piper = subprocess.Popen(
                [
                    PIPER_PATH,
                    "-m", PIPER_MODEL,
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

                stream.write(chunk)

            piper.wait()

        except Exception as e:

            log("TTS ERROR", str(e))

        finally:

            is_speaking = False
            speech_queue.task_done()

    stream.stop()
    stream.close()

threading.Thread(target=tts_worker, daemon=True).start()


# ============================================
# SPEECH THREAD
# ============================================

def speech_worker():

    global is_speaking

    while True:

        text = speech_queue.get()

        if text is None:
            break

        # THIS LOCK GUARANTEES NO OVERLAP
        with tts_lock:

            try:

                is_speaking = True

                log("TTS", f"Synthesizing: {text}")

                start = time.time()

                subprocess.run(
                    [
                        PIPER_PATH,
                        "-m", PIPER_MODEL,
                        "-f", OUTPUT_AUDIO
                    ],
                    input=text,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                log("TTS", f"Synth complete ({time.time()-start:.2f}s)")

                log("PLAYBACK", "Playing")

                winsound.PlaySound(
                    OUTPUT_AUDIO,
                    winsound.SND_FILENAME
                )

                log("PLAYBACK", "Done")

            except Exception as e:

                log("TTS ERROR", str(e))

            finally:

                is_speaking = False

        speech_queue.task_done()


threading.Thread(target=speech_worker, daemon=True).start()

# ============================================
# RECORD
# ============================================

def record_audio():

    log("RECORD", "Start")

    fs = 16000

    audio = sd.rec(
        int(RECORD_SECONDS * fs),
        samplerate=fs,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    write(INPUT_AUDIO, fs, audio)

    log("RECORD", "End")

# ============================================
# TRANSCRIBE
# ============================================

def transcribe():

    log("STT", "Start")

    start = time.time()

    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", INPUT_AUDIO,
            "-nt",
            "-t", str(CPU_THREADS)
        ],
        capture_output=True,
        text=True
    )

    text = re.sub(r'\[.*?\]', '', result.stdout).strip()

    log("STT", f"End ({time.time()-start:.2f}s): {text}")

    return text

# ============================================
# STREAM LLM (FIXED)
# ============================================

def ask_llm_stream(user_text):

    global is_generating

    is_generating = True

    log("LLM", "Streaming")

    prompt = f"""
WhatsApp automation sales agent.

Reply naturally in one short sentence.

Customer: {user_text}
Agent:
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True,
                "keep_alive": "24h",
                "options": {
                    "num_thread": CPU_THREADS,
                    "num_predict": 20,
                    "temperature": 0.3
                }
            },
            stream=True
        )

        buffer = ""

        for line in response.iter_lines():

            if not line:
                continue

            data = json.loads(line.decode())

            token = data.get("response", "")

            if token:
                log("LLM TOKEN", token)

            buffer += token

            # Extract ONLY complete sentences
            sentences = re.findall(r'[^.!?]*[.!?]', buffer)

            for sentence in sentences:

                clean = sentence.strip()

                if len(clean.split()) >= 2:

                    log("LLM SPEAK", clean)

                    speech_queue.put(clean)

            # Remove spoken sentences from buffer
            buffer = re.sub(r'[^.!?]*[.!?]', '', buffer)


        # speak remaining partial sentence if exists
        remaining = buffer.strip()

        if remaining:

            log("LLM FINAL", remaining)

            # speech_queue.put(remaining)
            if len(remaining.split()) >= 4:
                speech_queue.put(remaining)

        log("LLM", "Done")

    except Exception as e:

        log("LLM ERROR", str(e))

    finally:

        is_generating = False

# ============================================
# MAIN LOOP
# ============================================

def run_agent():

    log("SYSTEM", "VOICE AGENT READY")

    while True:

        try:

            if is_speaking or is_generating:
                time.sleep(0.05)
                continue

            record_audio()

            text = transcribe()

            if not text:
                continue

            log("USER", text)

            ask_llm_stream(text)

        except KeyboardInterrupt:

            log("SYSTEM", "Stopping")

            speech_queue.put(None)

            break

# ============================================
# START
# ============================================

if __name__ == "__main__":

    start_ollama()
    # start_piper()
    run_agent()
