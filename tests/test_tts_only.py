import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_agent.core.audio_handler import AudioHandler
from voice_agent.engines.tts import TTSEngine
from voice_agent.core.session import SessionState
from voice_agent.utils.logger import log

def test_tts():
    session = SessionState()
    audio = AudioHandler()
    
    # Use default devices
    print("--- TTS Single Test ---")
    log("SYSTEM", "Initializing audio...")
    stream = audio.create_output_stream()
    tts = TTSEngine(audio, session)
    
    message = "This is a long test message to verify that the TTS engine plays every single word until the very end without cutting off. If you hear this entire sentence, then the queue drain logic is working correctly."
    
    log("SYSTEM", "Queueing message...")
    tts.put(message)
    
    # Monitoring loop
    start_time = time.time()
    while True:
        is_speaking = session.is_speaking
        q_size = audio.out_queue.qsize()
        buffer_len = len(audio.out_buffer) if hasattr(audio, 'out_buffer') else 0
        
        print(f"\rStatus: Speaking={is_speaking}, Queue={q_size}, Buffer={buffer_len} bytes", end="")
        
        # If we've been running for 25 seconds, or it stopped speaking and queue is empty
        if time.time() - start_time > 25:
            print("\nTimeout reached.")
            break
        
        # Be more patient at the start (wait at least 5 seconds before checking "finished")
        if not is_speaking and q_size == 0 and buffer_len == 0 and time.time() - start_time > 5:
            print("\nPlayback finished or failed to start.")
            break
            
        time.sleep(0.1)

    log("SYSTEM", "Test compete. Shutting down.")
    stream.stop()
    stream.close()

if __name__ == "__main__":
    test_tts()
