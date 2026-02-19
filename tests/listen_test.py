import sounddevice as sd
from scipy.io.wavfile import write
import subprocess

WHISPER_PATH = r"C:\VOICE_AGENT\whisper\whisper-cli.exe"
MODEL_PATH = r"C:\VOICE_AGENT\whisper\models\ggml-base.en.bin"
AUDIO_FILE = r"C:\VOICE_AGENT\input.wav"

def record_audio(seconds=5):
    print("Speak now...")
    fs = 16000
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(AUDIO_FILE, fs, recording)
    print("Recording complete.")

def transcribe():
    result = subprocess.run(
        f'"{WHISPER_PATH}" -m "{MODEL_PATH}" -f "{AUDIO_FILE}"',
        shell=True,
        capture_output=True,
        text=True
    )
    print("You said:", result.stdout)

record_audio()
transcribe()
