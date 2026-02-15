import sounddevice as sd
import numpy as np
import queue
import threading
import time
import re
import requests
import json
import wave
import sys
import os
import signal
from datetime import datetime

# ==================== CONFIG ====================
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
SILENCE_TIMEOUT_MS = 1000
MIN_SPEECH_DURATION_MS = 300
ENERGY_THRESHOLD = 150                     # Lowered for sensitivity

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"                   # Use "phi3:mini-q4_0" for speed

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

# ==================== GLOBAL SHUTDOWN EVENT ====================
shutdown_event = threading.Event()

# ==================== TIMESTAMP DEBUG ====================
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")

# ==================== ENERGY-BASED VAD ====================
def is_speech(frame, threshold=ENERGY_THRESHOLD):
    rms = np.sqrt(np.mean(frame.astype(np.float32)**2))
    # Uncomment to calibrate threshold:
    # if rms > 10: log(f"RMS: {rms:.1f}")
    return rms > threshold

# ==================== STREAMING RECORDING ====================
def record_stream():
    log("Recording thread started")
    frames = []
    speech_start = None
    silence_frames = 0
    speech_frames_needed = int(MIN_SPEECH_DURATION_MS / CHUNK_DURATION_MS)
    silence_frames_needed = int(SILENCE_TIMEOUT_MS / CHUNK_DURATION_MS)

    def callback(indata, frames_count, time_info, status):
        nonlocal frames, speech_start, silence_frames
        if shutdown_event.is_set():
            raise sd.CallbackStop
        frame = indata[:, 0]
        if is_speech(frame):
            silence_frames = 0
            if speech_start is None:
                speech_start = time.time()
        else:
            silence_frames += 1
            if speech_start is not None and silence_frames > silence_frames_needed:
                if len(frames) >= speech_frames_needed:
                    audio_data = np.concatenate(frames)
                    audio_queue.put(audio_data)
                frames = []
                speech_start = None
                silence_frames = 0
        frames.append(frame.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16',
        blocksize=CHUNK_SIZE,
        callback=callback
    )
    with stream:
        while not shutdown_event.is_set():
            time.sleep(0.1)
    log("Recording thread stopped")

# ==================== STREAMING TRANSCRIPTION ====================
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    log("faster-whisper loaded")
    def transcribe_func(audio_np):
        segments, _ = model.transcribe(audio_np, beam_size=1, language="en")
        return " ".join([seg.text for seg in segments])
except ImportError:
    log("faster-whisper not installed, using whisper-cli fallback")
    import subprocess
    from scipy.io.wavfile import write
    WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
    WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-base.en.bin"
    def transcribe_func(audio_np):
        temp = "temp_input.wav"
        write(temp, SAMPLE_RATE, audio_np)
        result = subprocess.run(
            f'"{WHISPER_PATH}" -m "{WHISPER_MODEL}" -f "{temp}" -nt',
            shell=True, capture_output=True, text=True
        )
        raw = result.stdout.strip()
        if raw:
            text = re.sub(r'\[.*?\]', '', raw).strip()
        else:
            text = ""
        return text

def transcribe_stream():
    log("Transcription thread started")
    while not shutdown_event.is_set():
        try:
            audio_np = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        log("Transcribing...")
        text = transcribe_func(audio_np)
        if text:
            transcript_queue.put(text)
        else:
            log("No speech detected (empty transcription)")
    log("Transcription thread stopped")

# ==================== STREAMING LLM + TTS ====================
def ask_and_speak():
    global call_started, conversation_history
    log("LLM thread started")
    while not shutdown_event.is_set():
        try:
            user_text = transcript_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        log(f"User: {user_text}")

        conversation_history.append(f"Customer: {user_text}")
        conversation_history = conversation_history[-6:]

        if not call_started:
            instruction = "FIRST message. Greet and introduce."
            call_started = True
        else:
            instruction = "Continue naturally, no greeting."

        prompt = (
            "You are a sales agent. Be concise, max 2 sentences.\n"
            + instruction + "\n"
            + "\n".join(conversation_history)
            + "\nAgent:"
        )

        log("LLM streaming...")
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.6,
                        "num_predict": 60,
                        "num_ctx": 512
                    }
                },
                stream=True,
                timeout=(3, 10)  # connect timeout, read timeout
            )
        except Exception as e:
            log(f"LLM error: {e}")
            continue

        full_reply = []
        tts_buffer = ""

        for line in response.iter_lines():
            if shutdown_event.is_set():
                break
            if line:
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        full_reply.append(token)
                        tts_buffer += token
                        if any(punct in token for punct in ".!?"):
                            tts_queue.put(tts_buffer.strip())
                            tts_buffer = ""
                except:
                    pass

        if tts_buffer.strip() and not shutdown_event.is_set():
            tts_queue.put(tts_buffer.strip())

        reply = "".join(full_reply).strip()
        if not reply:
            reply = "Would you like to know more?"
        log(f"Agent: {reply}")
        conversation_history.append(f"Agent: {reply}")
    log("LLM thread stopped")

# ==================== STREAMING TTS PLAYBACK ====================
def tts_playback():
    log("TTS thread started")
    import subprocess as sp
    import pyaudio

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)

    piper_proc = sp.Popen(
        [PIPER_PATH, "-m", PIPER_MODEL, "--output_raw"],
        stdin=sp.PIPE,
        stdout=sp.PIPE,
        stderr=sp.DEVNULL,
        bufsize=0
    )

    def speak_chunk(text):
        piper_proc.stdin.write((text + "\n").encode())
        piper_proc.stdin.flush()
        # Read with timeout to avoid blocking forever
        BYTES_PER_SECOND = 22050 * 2
        bytes_to_read = int(BYTES_PER_SECOND * 1.5)
        data = b''
        start = time.time()
        while len(data) < bytes_to_read and not shutdown_event.is_set():
            available = piper_proc.stdout.read(min(4096, bytes_to_read - len(data)))
            if not available:
                break
            data += available
        stream.write(data)

    while not shutdown_event.is_set():
        try:
            text = tts_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        log(f"TTS: {text}")
        speak_chunk(text)

    # Cleanup
    piper_proc.terminate()
    piper_proc.wait()
    stream.stop_stream()
    stream.close()
    p.terminate()
    log("TTS thread stopped")

# ==================== QUEUES ====================
audio_queue = queue.Queue()
transcript_queue = queue.Queue()
tts_queue = queue.Queue()

# ==================== STATE ====================
conversation_history = []
call_started = False

# ==================== MAIN ====================
def main():
    log("Starting ultra-low-latency voice agent with energy VAD...")

    threads = [
        threading.Thread(target=record_stream, daemon=True),
        threading.Thread(target=transcribe_stream, daemon=True),
        threading.Thread(target=ask_and_speak, daemon=True),
        threading.Thread(target=tts_playback, daemon=True)
    ]
    for t in threads:
        t.start()

    try:
        # Wait for shutdown signal (Ctrl+C)
        while not shutdown_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        log("Ctrl+C received, shutting down...")
        shutdown_event.set()
    finally:
        # Give threads a moment to finish
        for t in threads:
            t.join(timeout=2)
        log("All threads stopped. Goodbye.")

if __name__ == "__main__":
    main()