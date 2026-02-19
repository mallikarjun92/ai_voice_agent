from faster_whisper import WhisperModel
import os
from voice_agent.utils.logger import log
from voice_agent.config import INPUT_AUDIO

class STTEngine:
    def __init__(self, model_size="tiny.en", device="cpu", compute_type="int8"):
        log("STT", f"Loading Faster-Whisper ({model_size})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        log("STT", "Ready")

    def transcribe(self, audio_path=INPUT_AUDIO):
        if not os.path.exists(audio_path):
            return ""
        
        segments, info = self.model.transcribe(audio_path, beam_size=1)
        text = " ".join([segment.text for segment in segments]).strip()
        if text:
            log("USER", text)
        return text
