import os

# ============================================
# PATHS & MODELS
# ============================================

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-tiny.en.bin"

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

INPUT_AUDIO = r"C:\VOICE_AGENT\input.wav"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:0.5b"

# ============================================
# SETTINGS
# ============================================

CPU_THREADS = 4
RECORD_SECONDS = 3
SILENCE_THRESHOLD = 500
SAMPLERATE = 22050

SYSTEM_PROMPT = "You are a Whatypie sales agent. Be human, warm, and extremely brief (max 1 sentence). Goal: Schedule demo."

STOP_TOKENS_MAP = {
    "gemma": ["<start_of_turn>", "<end_of_turn>", "user", "model", "User:", "Assistant:"],
    "qwen": ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "User:", "Assistant:"]
}

INSTANT_FILLERS = ["Got it.", "I see.", "Okay.", "Right.", "Mm-hmm.", "Uh-huh.", "Gotcha."]
