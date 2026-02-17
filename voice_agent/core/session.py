import threading

class SessionState:
    def __init__(self):
        self.is_speaking = False
        self.is_generating = False
        self.abort_speech = threading.Event()
        self.conversation = []
        self.speech_start_time = 0

    def abort(self):
        self.abort_speech.set()
        self.is_speaking = False
        self.is_generating = False

    def clear_abort(self):
        self.abort_speech.clear()
