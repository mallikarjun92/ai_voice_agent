import time
import random
import sys
import os

# Add parent directory to path to allow absolute imports if running from within the folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_agent.config import TARGET_NUMBER, INSTANT_FILLERS
from voice_agent.utils.logger import log
from voice_agent.core.session import SessionState
from voice_agent.core.audio_handler import AudioHandler
from voice_agent.core.telephony import TelephonyHandler
from voice_agent.engines.stt import STTEngine
from voice_agent.engines.llm import LLMEngine
from voice_agent.engines.tts import TTSEngine

def main():
    # 1. Initialize State & Handlers
    session = SessionState()
    audio = AudioHandler()
    telephony = TelephonyHandler(TARGET_NUMBER)
    
    # 2. Audio Setup (Hybrid Support)
    audio.list_devices()
    try:
        choice = input("\nSelect Audio Input Device ID (Enter for default): ").strip()
        if choice:
            audio.set_input_device(int(choice))
    except ValueError:
        log("SYSTEM", "Using default input device.")
    
    output_stream = audio.create_output_stream()
    
    # 3. Initialize Engines
    stt = STTEngine()
    llm = LLMEngine()
    tts = TTSEngine(output_stream, session)
    
    log("SYSTEM", "AGENT INITIALIZED & READY")
    
    # 4. Dialing Sequence
    log("SYSTEM", "Starting call sequence...")
    telephony.dial()
    
    # Wait for connection
    while not telephony.is_call_active():
        time.sleep(1)
    
    log("SYSTEM", "Call connected!")
    
    # First greeting
    tts.put("Hello, this is Mallikarjun from Whatypie. I wanted to quickly show how you can automate WhatsApp communication.")
    
    # 5. Main Conversation Loop
    try:
        while True:
            # Check if call still active
            if not telephony.is_call_active():
                log("SYSTEM", "Call ended by remote party.")
                break
            
            # Record user voice
            has_sound = audio.record(session)
            
            if has_sound:
                # Transcribe
                text = stt.transcribe()
                
                if text:
                    # Immediate acknowledgment
                    filler = random.choice(INSTANT_FILLERS)
                    tts.put(filler)
                    
                    # LLM Thinking & Response
                    llm.ask(text, session, tts.speech_queue)
                    
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        log("SYSTEM", "Shutting down...")
    finally:
        telephony.end_call()
        output_stream.stop()
        output_stream.close()

if __name__ == "__main__":
    main()
