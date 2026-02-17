import sounddevice as sd
import numpy as np
import sys
import os

# Add project root to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_agent.core.audio_handler import AudioHandler
from voice_agent.engines.tts import TTSEngine
from voice_agent.core.session import SessionState

def test_output():
    audio = AudioHandler()
    session = SessionState()
    
    print("--- TTS Output Tester ---")
    audio.list_devices()
    
    try:
        out_id = input("\nEnter Output Device ID to test: ").strip()
        if not out_id:
            print("No ID entered. Exiting.")
            return
        out_id = int(out_id)
        
        audio.set_output_device(out_id)
        stream = audio.create_output_stream()
        tts = TTSEngine(audio, session)
        
        print(f"\nPlaying test message to Device ID {out_id}...")
        print("Wait for it to speak...")
        
        tts.put("Testing audio output. If you can hear this, the routing is correct.")
        
        # Keep alive for a bit
        import time
        time.sleep(5)
        
        stream.stop()
        stream.close()
        print("\nTest finished.")
        
    except ValueError:
        print("Invalid ID. Use a number.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_output()
