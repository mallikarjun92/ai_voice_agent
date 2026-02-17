import subprocess
import time
from voice_agent.utils.logger import log

class TelephonyHandler:
    def __init__(self, target_number=None):
        self.target_number = target_number

    def dial(self, number=None):
        num = number or self.target_number
        if not num:
            log("SYSTEM", "No number provided for dialing")
            return
        
        log("SYSTEM", f"Dialing {num}...")
        subprocess.run([
            "adb", "shell", "am", "start", "-a", "android.intent.action.CALL",
            "-d", f"tel:{num}"
        ])
        time.sleep(2)
        self.enable_speaker()

    def enable_speaker(self):
        # Keyevent 164 is often Volume Mute, but sometimes toggles speaker in specific contexts/apps.
        # Robustness depends on the specific phone/dialer.
        log("SYSTEM", "Requesting speakerphone via ADB...")
        subprocess.run(["adb", "shell", "input", "keyevent", "164"])

    def is_call_active(self):
        try:
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "telephony.registry"],
                capture_output=True,
                text=True,
                timeout=2
            )
            # mCallState=2 usually means ACTIVE (Off-hook)
            # mCallState=1 is Ringing
            # mCallState=0 is Idle
            return "mCallState=2" in result.stdout
        except:
            return False

    def end_call(self):
        log("SYSTEM", "Ending call...")
        subprocess.run(["adb", "shell", "input", "keyevent", "KEYCODE_ENDCALL"])
