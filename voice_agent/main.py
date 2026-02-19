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
    
    # 2. Setup Mode
    print("\n--- Start Mode ---")
    print("1. Normal (Dial Phone)")
    print("2. Test Mode (No Dial, Default Audio)")
    mode_choice = input("Select Mode [1]: ").strip() or "1"
    
    is_test_mode = mode_choice == "2"
    
    if is_test_mode:
        log("SYSTEM", "TEST MODE: Skipping dial and using default audio devices.")
    else:
        # 2b. Audio Setup (Hybrid Support)
        audio.list_devices()
        try:
            in_choice = input("\nSelect Audio Input Device ID (Enter for default): ").strip()
            if in_choice:
                audio.set_input_device(int(in_choice))
            
            out_choice = input("Select Audio Output Device ID (Enter for default): ").strip()
            if out_choice:
                audio.set_output_device(int(out_choice))
        except ValueError:
            log("SYSTEM", "Using default devices where not specified.")
    
    output_stream = audio.create_output_stream()
    audio.start_background_recording()
    
    # ... Initialize Engines ...
    stt = STTEngine()
    llm = LLMEngine()
    tts = TTSEngine(audio, session)
    
    log("SYSTEM", "AGENT INITIALIZED & READY")
    
    if not is_test_mode:
        # 4. Dialing Sequence
        log("SYSTEM", "Starting call sequence...")
        telephony.dial()
        
        # Wait for connection
        while not telephony.is_call_active():
            time.sleep(1)
        
        log("SYSTEM", "Call connected!")
    
    # First greeting
    tts.put("Hello, this is Alex from Whatypie. I wanted to quickly show how you can automate WhatsApp communication.")
    
    # 5. Main Conversation Loop
    try:
        while True:
            # Check if call still active (only in Normal mode)
            if not is_test_mode and not telephony.is_call_active():
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
        if not is_test_mode:
            telephony.end_call()
        audio.stop_background_recording()
        output_stream.stop()
        output_stream.close()

if __name__ == "__main__":
    main()
