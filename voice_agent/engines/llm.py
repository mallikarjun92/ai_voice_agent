import requests
import json
import re
from voice_agent.utils.logger import log
from voice_agent.utils.text_processing import sanitize_text
from voice_agent.config import OLLAMA_URL, MODEL_NAME, SYSTEM_PROMPT, STOP_TOKENS_MAP, CPU_THREADS

class LLMEngine:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.preload()

    def preload(self):
        log("LLM", f"Preloading {self.model_name}...")
        try:
            requests.post(OLLAMA_URL, json={"model": self.model_name, "messages": [], "keep_alive": "24h"}, timeout=5)
            log("LLM", "Ready")
        except Exception as e:
            log("LLM", f"Preload failed: {e}")

    def ask(self, user_text, session, speech_queue):
        session.is_generating = True
        session.conversation.append({"role": "user", "content": user_text})
        
        # Keep last 4 turns for context
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.conversation[-4:]

        # Get stop tokens
        current_stops = []
        for key, tokens in STOP_TOKENS_MAP.items():
            if key in self.model_name.lower():
                current_stops = tokens
                break

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "num_thread": CPU_THREADS,
                        "num_predict": 50,
                        "temperature": 0.4,
                        "num_ctx": 2048,
                        "stop": current_stops + ["\nUser", "\nAssistant"]
                    },
                    "keep_alive": "5m"
                },
                stream=True
            )

            buffer = ""
            full_response = []

            for line in response.iter_lines():
                if session.abort_speech.is_set():
                    log("LLM", "Aborted")
                    break
                if not line: continue
                
                token = json.loads(line.decode())["message"]["content"]
                
                # Hallucination Guard
                lower_token = token.lower()
                active_guards = [s.lower() for s in current_stops if len(s) > 2]
                if any(x in lower_token for x in active_guards):
                    log("SYSTEM", f"Hallucination Guard Triggered: {token}")
                    break

                buffer += token

                # Sentence-Based Streaming
                if any(p in buffer for p in ".!?"):
                    match = re.search(r'(.*[.!?])', buffer)
                    if match:
                        sentence = match.group(0).strip()
                        if sentence:
                            clean_sent = sanitize_text(sentence)
                            if clean_sent:
                                log("AGENT", f"(Stream) {clean_sent}")
                                speech_queue.put(clean_sent)
                                full_response.append(clean_sent)
                        
                        buffer = buffer[len(match.group(0)):]
                        continue

            if buffer.strip() and not session.abort_speech.is_set():
                clean_last = sanitize_text(buffer.strip())
                if clean_last:
                    log("AGENT", f"(Final) {clean_last}")
                    speech_queue.put(clean_last)
                    full_response.append(clean_last)

            if full_response:
                session.conversation.append({"role": "assistant", "content": " ".join(full_response)})

        except Exception as e:
            log("LLM", f"Error: {e}")
        finally:
            session.is_generating = False
