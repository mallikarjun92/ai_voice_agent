import subprocess

PIPER_PATH = r"C:\VOICE_AGENT\piper\piper.exe"
MODEL_PATH = r"C:\VOICE_AGENT\piper\models\en_US-lessac-medium.onnx"
OUTPUT_FILE = r"C:\VOICE_AGENT\output.wav"

def speak(text):
    # Generate speech
    subprocess.run(
        f'echo {text} | "{PIPER_PATH}" -m "{MODEL_PATH}" -f "{OUTPUT_FILE}"',
        shell=True
    )

    # Play speech using Windows PowerShell
    subprocess.run(
        f'powershell -c (New-Object Media.SoundPlayer "{OUTPUT_FILE}").PlaySync();',
        shell=True
    )

speak("Hello Mallikarjun, your autonomous AI agent is now speaking.")
