import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import requests
import time

# ======================
# PATH CONFIGURATION
# ======================

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
WHISPER_MODEL = r"C:\VOICE_AGENT\whisper\models\ggml-base.en.bin"

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
PIPER_MODEL = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"

INPUT_AUDIO = r"C:\VOICE_AGENT\input.wav"
OUTPUT_AUDIO = r"C:\VOICE_AGENT\output.wav"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma:2b"

# ======================
# SYSTEM PROMPT
# ======================

SYSTEM_PROMPT = """
You are a human sales agent on a live phone call selling WhatsApp automation software.

The call is already connected and ongoing.

IMPORTANT RULES:
- NEVER greet again after the first message
- NEVER say Hello, Hi, Hey, or greetings again
- Only greet in the very first message of the entire conversation
- After that, respond directly to the customer's question
- Speak like a real human on a continuous phone call
- Be concise (max 1–2 sentences)
- Focus on explaining benefits and booking a demo

Bad examples:
Customer: What is it?
Agent: Hello! It is a software...
(This is WRONG)

Good example:
Customer: What is it?
Agent: It automates your WhatsApp replies and helps you convert more leads automatically.
"""

# ======================
# MEMORY
# ======================

conversation_history = []


# ======================
# RECORD AUDIO
# ======================

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


# ======================
# TRANSCRIBE AUDIO
# ======================

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

    # Clean timestamps if present
    lines = raw.split("\n")
    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if "-->" in line:
            parts = line.split("]")
            if len(parts) > 1:
                cleaned_lines.append(parts[1].strip())
        else:
            cleaned_lines.append(line)

    text = " ".join(cleaned_lines).strip()

    return text


# ======================
# ASK LLM
# ======================

def ask_llm(user_text):

    global conversation_history

    if not user_text.strip():
        return "Could you please repeat that?"

    # Add user message
    conversation_history.append(f"Customer: {user_text}")

    # Limit memory
    conversation_history = conversation_history[-6:]

    prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(conversation_history) + "\nAgent:"

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "num_predict": 60,
                    "temperature": 0.7,
                    "stop": ["Customer:", "Agent:"]
                }
            },
            timeout=60
        )

        reply = response.json().get("response", "").strip()

    except Exception as e:

        print("LLM Error:", e)
        reply = ""

    # fallback protection
    if not reply:
        reply = "I'm calling to help automate your WhatsApp business communication. Would you like a quick demo?"

    # Save reply
    conversation_history.append(f"Agent: {reply}")

    return reply


# ======================
# SPEAK TEXT
# ======================

def speak(text):

    if not text.strip():
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


# ======================
# MAIN LOOP
# ======================

def run_agent():

    print("\n====================================")
    print(" AI Voice Agent Started")
    print(" Press Ctrl+C to stop")
    print("====================================")

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

            print("\n--- Conversation History ---")
            print("\n".join(conversation_history))
            print("----------------------------")

            time.sleep(0.3)

        except KeyboardInterrupt:

            print("\nAgent stopped.")
            break

        except Exception as e:

            print("Error:", e)
            time.sleep(1)


# ======================
# START AGENT
# ======================

if __name__ == "__main__":

    run_agent()
