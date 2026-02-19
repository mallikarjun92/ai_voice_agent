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
            # Check telephony registry for precise state
            result = subprocess.run(
                ["adb", "shell", "dumpsys", "telephony.registry"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=2
            )
            stdout = result.stdout or ""

            # mCallState=2 is Off-hook (broad)
            # mPreciseCallState=3 is ACTIVE (specific)
            # Some devices use 'mForegroundCallState=1' for active
            
            is_active = "mPreciseCallState=3" in stdout or "mForegroundCallState=1" in stdout
            
            # Fallback to broader check if precise isn't found, but maybe with a delay
            if not is_active and "mCallState=2" in stdout:
                # If we only have mCallState=2, we might still be dialing.
                # We'll check dumpsys telecom for 'State: ACTIVE' as a second opinion.
                telecom_result = subprocess.run(
                    ["adb", "shell", "dumpsys", "telecom"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='ignore',
                    timeout=2
                )
                stdout_telecom = telecom_result.stdout or ""
                is_active = "State: ACTIVE" in stdout_telecom

            return is_active
        except Exception as e:
            log("SYSTEM", f"Connection check error: {e}")
            return False

    def end_call(self):
        log("SYSTEM", "Ending call...")
        subprocess.run(["adb", "shell", "input", "keyevent", "KEYCODE_ENDCALL"])
