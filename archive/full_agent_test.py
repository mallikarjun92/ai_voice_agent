import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import requests
import time

# ============================================
# CONFIGURATION
# ============================================

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-base.en.bin"

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

INPUT_AUDIO = r"C:\VOICE_AGENT\input.wav"
OUTPUT_AUDIO = r"C:\VOICE_AGENT\output.wav"

OLLAMA_URL = "http://localhost:11434/api/generate"

# Recommended models:
# gemma:2b (fast)
# phi3:mini (better quality)
MODEL_NAME = "gemma:2b"

# ============================================
# SYSTEM PROMPT (STRONG VERSION)
# ============================================

SYSTEM_PROMPT = """
You are a real human sales agent calling a business owner to offer WhatsApp automation software.

This is a LIVE ongoing phone call.

CRITICAL RULES:

- Only greet ONCE at the very beginning
- NEVER greet again
- NEVER say Hello, Hi, Hey again after first message
- Continue conversation naturally
- Answer questions directly
- Keep responses short (max 2 sentences)
- Sound human, confident, and professional
- Focus on explaining benefits and booking demo

BAD:
Customer: What is it?
Agent: Hello! It is software...

GOOD:
Customer: What is it?
Agent: It automates your WhatsApp replies and helps convert more leads automatically.
"""

# ============================================
# MEMORY STATE
# ============================================

conversation_history = []
greeted = False


# ============================================
# REMOVE REPEATED GREETING
# ============================================

def remove_repeat_greeting(reply):

    global greeted

    greetings = [
        "hello", "hi", "hey",
        "hello!", "hi!", "hey!",
        "hello there", "hi there", "hey there"
    ]

    words = reply.lower().split()

    if greeted and len(words) > 0:

        first_word = words[0]

        if first_word in greetings:

            reply = " ".join(reply.split()[1:])

    return reply.strip()


# ============================================
# RECORD AUDIO
# ============================================

def record_audio(seconds=4):

    fs = 16000

    print("\nListening...")

    recording = sd.rec(
        int(seconds * fs),
        samplerate=fs,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    write(INPUT_AUDIO, fs, recording)


# ============================================
# TRANSCRIBE AUDIO
# ============================================

def transcribe():

    result = subprocess.run(
        f'"{WHISPER_PATH}" -m "{WHISPER_MODEL}" -f "{INPUT_AUDIO}" -nt',
        shell=True,
        capture_output=True,
        text=True
    )

    raw = result.stdout.strip()

    if not raw:
        return ""

    lines = raw.split("\n")

    cleaned = []

    for line in lines:

        line = line.strip()

        if "-->" in line:
            parts = line.split("]")
            if len(parts) > 1:
                cleaned.append(parts[1].strip())
        else:
            cleaned.append(line)

    text = " ".join(cleaned).strip()

    return text


# ============================================
# ASK LLM
# ============================================

def ask_llm(user_text):

    global conversation_history, greeted

    if not user_text:
        return "Could you please repeat that?"

    conversation_history.append(f"Customer: {user_text}")

    # limit memory
    conversation_history = conversation_history[-6:]

    if not greeted:

        greeting_instruction = "Start with greeting and introduce yourself."

        greeted = True

    else:

        greeting_instruction = "DO NOT greet. Continue conversation naturally."

    prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + greeting_instruction
        + "\n\n"
        + "\n".join(conversation_history)
        + "\nAgent:"
    )

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.7,
                    "num_predict": 60,
                    "stop": ["Customer:", "Agent:"]
                }
            },
            timeout=60
        )

        reply = response.json().get("response", "").strip()

    except Exception as e:

        print("LLM error:", e)

        reply = ""

    # fallback
    if not reply:

        reply = "I'm calling to help automate your WhatsApp customer communication. Would you like to know more?"

    # remove repeated greeting
    reply = remove_repeat_greeting(reply)

    conversation_history.append(f"Agent: {reply}")

    return reply


# ============================================
# SPEAK RESPONSE
# ============================================

def speak(text):

    if not text:
        return

    try:

        subprocess.run(
            f'echo {text} | "{PIPER_PATH}" -m "{PIPER_MODEL}" -f "{OUTPUT_AUDIO}"',
            shell=True
        )

        subprocess.run(
            f'powershell -c (New-Object Media.SoundPlayer "{OUTPUT_AUDIO}").PlaySync();',
            shell=True
        )

    except Exception as e:

        print("Speech error:", e)


# ============================================
# MAIN LOOP
# ============================================

def run_agent():

    print("\n===================================")
    print(" AUTONOMOUS AI VOICE AGENT STARTED")
    print(" Press Ctrl+C to stop")
    print("===================================")

    while True:

        try:

            record_audio()

            user_text = transcribe()

            if not user_text:
                print("No speech detected")
                continue

            print("\nCustomer:", user_text)

            reply = ask_llm(user_text)

            print("Agent:", reply)

            speak(reply)

            print("\n--- Conversation ---")
            print("\n".join(conversation_history))
            print("--------------------")

            time.sleep(0.3)

        except KeyboardInterrupt:

            print("\nAgent stopped.")

            break

        except Exception as e:

            print("Error:", e)

            time.sleep(1)


# ============================================
# START
# ============================================

if __name__ == "__main__":

    run_agent()
