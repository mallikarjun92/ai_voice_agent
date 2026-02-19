# AI Voice Agent: User Guide

Welcome to the AI Voice Agent. This guide will help you get started with your first automated call.

## 📋 Prerequisites

Before running the agent, ensure you have:
1.  **Android Phone**: Connected via USB with **USB Debugging** enabled.
2.  **Ollama**: Running locally with the `qwen2.5:0.5b` model pulled.
3.  **Sndcpy**: (Optional but recommended) For direct audio forwarding.

## 🚀 Quick Start

### 1. Update Configuration
Open `voice_agent/config.py` and set your test number:
```python
TARGET_NUMBER = "+91XXXXXXXXXX"
```

### 2. Start Audio Forwarding (Recommended)
If you want to hear the phone's audio directly on your PC:
1.  Run `sndcpy` in a separate terminal.
2.  Press "Start now" on your phone.

### 3. Launch the Agent
Run the following command from the project root:
```powershell
python -m voice_agent.main
```

### 4. Select Audio Input
- The system will list available devices.
- Look for the device labeled **"CABLE Output"** or similar (if using Sndcpy/Virtual Cable) and enter its ID.
- If testing locally with a microphone, just press **Enter**.

## 📞 During the Call

- **The agent is autonomous**: It will dial, wait for a connection, and greet the recipient.
- **Natural Conversation**: You can speak to the agent. It uses Voice Activity Detection (VAD) to know when you've finished talking.
- **Interruptions**: You can interrupt the agent while it is speaking. It will stop and listen to you immediately.

## ⏹️ Stopping the Agent

- To end the call and stop the agent, press `Ctrl + C` in the terminal.
- The agent will automatically hang up the phone.

---
*For technical details, see the [System Guide](system_guide.md).*
