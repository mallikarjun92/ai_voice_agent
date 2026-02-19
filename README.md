# AI Voice Agent (Autonomous Calling)

A modular, real-time AI voice agent designed to handle automated outbound calls via ADB (Android Debug Bridge). It leverages state-of-the-art local AI models for speech processing and conversation.

## 🚀 Features

- **Hybrid Audio Selection**: Supports both hardware microphones (proximity testing) and virtual audio cables (direct digital routing via Sndcpy).
- **Modular Architecture**: Separated core logic, AI engines, and utilities for high maintainability.
- **Low Latency Processing**: Sentence-based streaming for LLM responses and TTS generation.
- **Autonomous Call Handling**: Automatic dialing, connection monitoring, and interruption detection.
- **Hallucination Guards**: Dynamic stop-token monitoring to prevent LLM loops.

---

## 📂 Project Structure

```text
voice_agent/
├── main.py                 # Orchestrator & entry point
├── config.py               # Global constants, paths, and model settings
├── core/
│   ├── audio_handler.py    # Hybrid Audio Manager (Input/Output streams)
│   ├── telephony.py        # ADB Call control & state monitoring
│   └── session.py          # Shared state & interruption manager
├── engines/
│   ├── stt.py              # Speech-to-Text (Faster-Whisper)
│   ├── tts.py              # Text-to-Speech (Piper)
│   └── llm.py              # LLM Interface (Ollama/Qwen)
└── utils/
    ├── logger.py           # Standardized timestamped logging
    └── text_processing.py  # Sanitization & sentence splitting
```

---

## 🛠️ Prerequisites

Ensure the following tools are installed and configured in `config.py`:

1.  **ADB (Android Debug Bridge)**: Must be in your system PATH and the phone must be connected with USB Debugging enabled.
2.  **Ollama**: Install [Ollama](https://ollama.com/) and pull the `qwen2.5:0.5b` model.
3.  **Piper TTS**: Download [Piper](https://github.com/rhasspy/piper) and a `.onnx` model.
4.  **Faster-Whisper**: Install via pip: `pip install faster-whisper`.
5.  **SoundDevice & SciPy**: Install for audio I/O: `pip install sounddevice scipy numpy`.

---

## 📖 Usage

### 1. Configuration
Open `config.py` and update the paths for Piper and Whisper, as well as the `TARGET_NUMBER`.

### 2. Execution
Run the agent from the root directory:
```powershell
python -m voice_agent.main
```

### 3. Audio Selection
Upon startup, the script will list all available audio devices.
- **For direct phone-to-PC digital audio**: Connect via `sndcpy` and enter the Device ID of the virtual sink.
- **For local testing**: Just press **Enter** to use your default system microphone.

---

## ⚠️ Known Issues & Edge Cases

- **Speakerphone**: ADB `keyevent 164` behavior varies by Android dialer. Ensure your phone's dialer application supports this intent or manually toggle speakerphone if needed.
- **Feedback Loops**: When using hardware microphones near speakers, ensure volume is low or use headphones to avoid transcribed "echoes".
- **Voicemail**: The agent currently handles voicemails by transcribing them. Future updates will include automated voicemail detection/hangup.

---

## 📜 Development
This project was refactored from a monolithic script into a modular package to support autonomous call features and a hybrid audio environment. 
