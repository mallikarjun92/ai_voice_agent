import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import requests
import time
import re

# ============================================
# CONFIG
# ============================================

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-base.en.bin"

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

INPUT_AUDIO = r"C:\VOICE_AGENT\input.wav"
OUTPUT_AUDIO = r"C:\VOICE_AGENT\output.wav"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"   # better conversational behavior than gemma

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
# STATE
# ============================================

conversation_history = []
call_started = False


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
# TRANSCRIBE
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

    # clean timestamps
    text = re.sub(r'\[.*?\]', '', raw).strip()

    return text


# ============================================
# CHECK IF INPUT IS UNCLEAR
# ============================================

def is_unclear(text):

    if len(text) < 3:
        return True

    unclear_words = [
        "okay",
        "ok",
        "hmm",
        "uh",
        "ah",
        "yes",
        "no",
        "peace"
    ]

    if text.lower().strip() in unclear_words:
        return True

    return False


# ============================================
# ASK LLM
# ============================================

def ask_llm(user_text):

    global conversation_history, call_started

    # clarification handling
    if is_unclear(user_text):

        return "Sorry, could you please clarify that?"

    conversation_history.append(f"Customer: {user_text}")

    conversation_history = conversation_history[-8:]

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

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.6,
                    "num_predict": 80
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

        if not call_started:

            reply = "Hello, this is Alex calling about WhatsApp automation for your business. Is this a good time?"

            call_started = True

        else:

            reply = "Would you like to know how it can help your business?"

    conversation_history.append(f"Agent: {reply}")

    return reply


# ============================================
# SPEAK
# ============================================

def speak(text):

    if not text:
        return

    subprocess.run(
        f'echo {text} | "{PIPER_PATH}" -m "{PIPER_MODEL}" -f "{OUTPUT_AUDIO}"',
        shell=True
    )

    subprocess.run(
        f'powershell -c (New-Object Media.SoundPlayer "{OUTPUT_AUDIO}").PlaySync();',
        shell=True
    )


# ============================================
# MAIN LOOP
# ============================================

def run_agent():

    print("\nAI CALLING AGENT STARTED")

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

            print("\nConversation:")
            print("\n".join(conversation_history))

        except KeyboardInterrupt:

            print("\nStopped.")

            break

        except Exception as e:

            print("Error:", e)

            time.sleep(1)


# ============================================

if __name__ == "__main__":

    run_agent()
