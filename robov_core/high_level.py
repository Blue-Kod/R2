import os
import platform
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

import cv2
import numpy as np
import psutil

from robov_core.camera import StereoCamera
from robov_core.servo import ServoController

try:
    from robov_core.eyes_display import EyeDisplay, optimize_for_arm
    _HAS_DISPLAY = True
except ImportError:
    _HAS_DISPLAY = False
    EyeDisplay = None
    optimize_for_arm = None

# --- Configuration ---
APP_VERSION = "0.2"
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 80
CAMERA_SOURCE = 0
CAMERA_PARAMS_FILE = "cam_params.json"
LAUNCHER_SCRIPT = "launcher.py"
EYES_SCALE_FACTOR = 1.3
APP_PASSWORD = "admin."

DEFAULT_SERVO_ANGLES: Dict[int, int] = {
    0: 90, 1: 135, 2: 135, 3: 90, 4: 45,
    5: 45, 6: 135, 7: 135, 8: 90, 9: 90
}

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Global state ---
_camera: Optional[StereoCamera] = None
_servo: Optional[ServoController] = None
_display: Optional[EyeDisplay] = None
_eye_api = None
_lock: threading.Lock = threading.Lock()
_servo_lock: threading.Lock = threading.Lock()
_display_lock: threading.Lock = threading.Lock()

_logs_buffer: deque = deque(maxlen=500)


class StdoutCapture:
    def __init__(self) -> None:
        self._original_stdout = sys.stdout
        self._lock: threading.Lock = threading.Lock()

    def write(self, message: str) -> None:
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        self._original_stdout.write(message)
        self._original_stdout.flush()
        if message.strip():
            with self._lock:
                for line in message.strip().split('\n'):
                    if line.strip():
                        _logs_buffer.append(line)

    def flush(self) -> None:
        self._original_stdout.flush()


_stdout_capture = StdoutCapture()
sys.stdout = _stdout_capture
sys.stderr = _stdout_capture

_hardware_initialized: bool = False
_shutdown_requested: bool = False

_shell_proc = None
_shell_buffer: deque = deque(maxlen=2000)
_shell_running: bool = False
_shell_lock: threading.Lock = threading.Lock()
_shell_thread: Optional[threading.Thread] = None

_current_emote: str = "normal"
_eyes_x: float = 0.0
_eyes_y: float = 0.0
_emote_lock: threading.Lock = threading.Lock()
_emotions_dir: str = os.path.join(os.path.dirname(__file__), "emotions")
_supported_emotes: List[str] = [os.path.splitext(f)[0] for f in os.listdir(_emotions_dir) if f.endswith(".png")]

_display_thread: Optional[threading.Thread] = None
_all_threads: List[threading.Thread] = []

_tts: Any = None
_tts_sample_rate: int = 22050
_tts_lock: threading.Lock = threading.Lock()

_ai_agent = None

_TTS_MODEL_DIR = os.path.join(ROOT_DIR, "models")
_TTS_MODEL_NAME = "ru_RU-ruslan-medium"
_TTS_LENGTH_SCALE = 0.9


def _init_tts():
    """Load Piper TTS model with optimized ONNX session."""
    global _tts, _tts_sample_rate
    try:
        import onnxruntime
        from piper import PiperVoice
        onnx_path = os.path.join(_TTS_MODEL_DIR, f"{_TTS_MODEL_NAME}.onnx")
        if not os.path.exists(onnx_path):
            log("TTS model not found, attempting download...")
            _download_tts_model()
        _tts = PiperVoice.load(onnx_path)
        _tts_sample_rate = _tts.config.sample_rate

        cpu_count = os.cpu_count() or 4
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.intra_op_num_threads = cpu_count
        sess_opts.inter_op_num_threads = max(1, cpu_count // 2)
        sess_opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        _tts.session = onnxruntime.InferenceSession(
            onnx_path, sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        import io, wave
        with wave.open(io.BytesIO(), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_tts_sample_rate)
            _tts.synthesize_wav("ок", w)
        log(f"Piper TTS ready (sample_rate={_tts_sample_rate}, threads={cpu_count})")
    except ImportError:
        log("TTS unavailable: piper-tts not installed")
    except Exception as e:
        log(f"TTS init error: {e}")
        _tts = None


def speak(text: str) -> None:
    global _tts
    if _tts is None:
        _init_tts()
    if _tts is None:
        return
    try:
        import subprocess, io, wave
        aplay = subprocess.Popen(
            ["aplay", "-D", "plughw:1,0", "-t", "raw",
             "-f", "S16_LE", "-r", str(_tts_sample_rate), "-c", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for chunk in _tts.synthesize(text):
            aplay.stdin.write(chunk.audio_int16_bytes)
        aplay.stdin.close()
        aplay.wait()
    except Exception as e:
        log(f"TTS error: {e}")


def _download_tts_model():
    """Download Piper TTS model from HuggingFace if missing."""
    import requests as _req
    hf_base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/ruslan/medium"
    os.makedirs(_TTS_MODEL_DIR, exist_ok=True)
    for ext in ("onnx", "onnx.json"):
        url = f"{hf_base}/{_TTS_MODEL_NAME}.{ext}"
        dest = os.path.join(_TTS_MODEL_DIR, f"{_TTS_MODEL_NAME}.{ext}")
        if os.path.exists(dest):
            continue
        try:
            log(f"Downloading {_TTS_MODEL_NAME}.{ext}...")
            r = _req.get(url, stream=True, timeout=120)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(256 * 1024):
                    f.write(chunk)
            log(f"Saved {_TTS_MODEL_NAME}.{ext}")
        except Exception as e:
            log(f"Failed to download {ext}: {e}")

_MIN_WORD_LEN = 5

class StreamingSpeaker:
    """Buffers LLM tokens and speaks complete word-chunks in a background thread."""

    def __init__(self):
        self._buf = ""
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, args=(self._queue,), daemon=True)
        self._thread.start()

    def feed(self, text: str) -> None:
        self._buf += text
        self._try_split()

    def flush(self) -> None:
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._queue.put(rest)
        self._queue.put(None)
        self._thread.join()

    def reset(self) -> None:
        self._buf = ""
        self._queue.put(None)
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, args=(self._queue,), daemon=True)
        self._thread.start()

    def _try_split(self) -> None:
        while True:
            if " " not in self._buf:
                return
            last_space = self._buf.rfind(" ")
            prefix = self._buf[:last_space]
            if not prefix:
                return
            words = prefix.split()
            if any(len(w) >= _MIN_WORD_LEN for w in words):
                self._queue.put(prefix.strip())
                self._buf = self._buf[last_space + 1:]
                continue
            return

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"`[^`]+`", "", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        text = re.sub(r"~~([^~]+)~~", r"\1", text)
        text = re.sub(r"#+ ", "", text)
        text = re.sub(r"[-*] ", "", text)
        return text.strip()

    @staticmethod
    def _worker(q: queue.Queue) -> None:
        while True:
            chunk = q.get()
            if chunk is None:
                break
            cleaned = StreamingSpeaker._clean(chunk)
            if cleaned:
                speak(cleaned)


def log(message: str) -> None:
    print(message)


def _init_hardware() -> None:
    global _camera, _servo

    cam_path = ROOT_DIR / CAMERA_PARAMS_FILE
    if cam_path.exists():
        try:
            _camera = StereoCamera(str(cam_path), source=CAMERA_SOURCE)
            if _camera.initialize_camera():
                _camera.start_continuous_capture()
                log(f"Camera initialized on {platform.system()}")
            else:
                log("Camera capture failed - continuing with mock camera")
        except Exception as exc:
            log(f"Camera init error: {exc} - continuing with mock camera")
    else:
        log("Camera config not found - continuing with mock camera")

    if platform.system() == "Windows":
        log("Mock servo mode on Windows")
    else:
        try:
            subprocess.run(
                ["amixer", "-c", "1", "cset", "numid=18", "191"],
                capture_output=True, timeout=5
            )
            subprocess.run(
                ["amixer", "-c", "1", "cset", "numid=19", "191"],
                capture_output=True, timeout=5
            )
            log("Audio volume set to ~75% (DACL/DACR = 191)")
        except Exception as exc:
            log(f"Volume set failed: {exc}")

        try:
            _servo = ServoController(bus=0, address=0x40, freq=50)
            log("Servo controller initialized")
            for channel, angle in DEFAULT_SERVO_ANGLES.items():
                if channel in _servo.channel_configs:
                    _servo.set_servo(channel, angle, smooth=False)
        except Exception as exc:
            log(f"Servo init error: {exc}")
            log("Falling back to mock servo")


def _shell_reader() -> None:
    global _shell_running
    try:
        while _shell_running:
            if _shell_proc is None:
                data = b""
            elif platform.system() == "Windows":
                data = _shell_proc.stdout.read1(1024) if _shell_proc.stdout else b""
            else:
                import ptyprocess
                data = _shell_proc.read(1024) if hasattr(_shell_proc, 'read') else b""
            if not data:
                break
            text = _decode_output(data)
            with _shell_lock:
                _shell_buffer.append(text)
    except Exception:
        pass
    finally:
        _shell_running = False


def _decode_output(data: bytes) -> str:
    for enc in ("utf-8", "cp866", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def shell_start() -> None:
    global _shell_proc, _shell_running, _shell_thread

    if _shell_running:
        return
    _shell_running = True

    if platform.system() == "Windows":
        try:
            _shell_proc = subprocess.Popen(
                ["powershell", "-NoLogo"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            log(f"Failed to start PowerShell: {exc}")
            _shell_running = False
            return
    else:
        try:
            import ptyprocess
            _shell_proc = ptyprocess.PtyProcess.spawn(["/bin/bash", "-i"])
            _shell_proc.setwinsize(24, 80)
        except Exception as exc:
            log(f"Failed to start shell: {exc}")
            _shell_running = False
            return

    _shell_thread = threading.Thread(target=_shell_reader, daemon=True)
    _shell_thread.start()


def shell_write(command: str) -> bool:
    if not _shell_running or _shell_proc is None:
        return False
    try:
        if not command.endswith("\n"):
            command += "\n"
        if platform.system() == "Windows":
            if not _shell_proc.stdin:
                return False
            _shell_proc.stdin.write(command.encode("utf-8"))
            _shell_proc.stdin.flush()
        else:
            _shell_proc.write(command.encode("utf-8"))
        return True
    except Exception:
        return False


def shell_output() -> str:
    with _shell_lock:
        return "".join(_shell_buffer)


def shell_onetime(command: str) -> str:
    old_output = shell_output()
    shell_write(command)
    return shell_output().replace(old_output, "")


def set_emote(emotion_name: str) -> bool:
    global _current_emote
    name = str(emotion_name or "").strip().lower()
    if name not in _supported_emotes:
        return False
    with _emote_lock:
        _current_emote = name
    if _eye_api:
        try:
            _eye_api.update_emote(name)
        except Exception:
            pass
    return True


def get_emote() -> str:
    with _emote_lock:
        return _current_emote


def set_eyes_position(x: float, y: float) -> None:
    global _eyes_x, _eyes_y
    x = max(-1.0, min(1.0, float(x)))
    y = max(-1.0, min(1.0, float(y)))
    with _emote_lock:
        _eyes_x = x
        _eyes_y = y
    if _eye_api:
        try:
            _eye_api.update_eyes_position(x, y)
        except Exception:
            pass


def get_eyes_position() -> Tuple[float, float]:
    with _emote_lock:
        return _eyes_x, _eyes_y


def supported_emotes() -> List[str]:
    return sorted(_supported_emotes)


def cpu_temp() -> str:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as file:
            temp = int(file.read()) / 1000
            return f"{temp:.1f}°C"
    except Exception:
        return "N/A"


def ip_address() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def get_logs(n: int = 500) -> List[str]:
    with _lock:
        return list(_logs_buffer)[-n:]


def health_snapshot() -> dict:
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "temp": cpu_temp(),
    }


def _display_worker() -> None:
    global _display, _eye_api

    if not _HAS_DISPLAY:
        print("[!] eyes_display.py not available, skipping display")
        return

    if optimize_for_arm:
        optimize_for_arm()

    with _display_lock:
        _display = EyeDisplay(scale_factor=EYES_SCALE_FACTOR)

    _eye_api = _display.api
    _display.start()


def stop_display() -> None:
    global _display, _eye_api, _display_thread
    if _display:
        try:
            _display.stop()
        except Exception:
            pass
        _display = None
        _eye_api = None
        _display_thread = None


def start_display() -> None:
    global _display, _eye_api, _display_thread
    if _display_thread and _display_thread.is_alive():
        return
    if not _HAS_DISPLAY:
        return
    t = threading.Thread(target=_display_worker, daemon=True, name="r2-display-thread")
    t.start()
    _display_thread = t
    _all_threads.append(t)


# --- Shell ---


def start_background() -> None:
    global _hardware_initialized, _all_threads

    if _hardware_initialized:
        return

    log(f"R2 v{APP_VERSION} - Starting...")

    _init_hardware()
    shell_start()
    if _shell_thread:
        _all_threads.append(_shell_thread)

    from robov_core.web import create_app
    app = create_app()
    web_thread = threading.Thread(
        target=lambda: app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True, use_reloader=False),
        daemon=True,
        name="r2-web-thread"
    )
    web_thread.start()
    _all_threads.append(web_thread)

    # WebSocket terminal server disabled — terminal uses polling via REST API

    if _HAS_DISPLAY:
        global _display_thread
        _display_thread = threading.Thread(target=_display_worker, daemon=True, name="r2-display-thread")
        _display_thread.start()
        _all_threads.append(_display_thread)

    # Initialize AI agent
    try:
        from robov_core.ai import init_agent
        global _ai_agent
        _ai_agent = init_agent(cwd=str(ROOT_DIR))
        # Inject robot API into AI's python environment
        _ai_agent.executor.python_env.update({
            "angle": angle,
            "emote": emote,
            "speak": speak,
            "set_emote": set_emote,
            "get_emote": get_emote,
            "set_eyes_position": set_eyes_position,
            "get_eyes_position": get_eyes_position,
            "get_stereo_camera": get_stereo_camera,
            "get_raw_frame": get_raw_frame,
            "get_coords_stereo": get_coords_stereo,
            "health_snapshot": health_snapshot,
            "ip_address": ip_address,
            "log": log,
            "cleanup": cleanup,
            "shell_start": shell_start,
            "shell_write": shell_write,
            "shell_output": shell_output,
            "shell_onetime": shell_onetime,
            "get_servo_angles": get_servo_angles,
            "get_servo_offsets": get_servo_offsets,
            "set_servo_physical": set_servo_physical,
            "get_servo_angles_physical": get_servo_angles_physical,
            "set_emote": set_emote,
            "supported_emotes": supported_emotes,
        })
        log("AI agent initialized")
    except Exception as e:
        log(f"AI agent init error: {e}")

    _hardware_initialized = True


def cleanup() -> None:
    global _shutdown_requested, _hardware_initialized, _shell_running

    if not _hardware_initialized:
        return

    log("Starting clean shutdown...")
    _shutdown_requested = True
    _shell_running = False

    if _camera:
        _camera.stop_continuous_capture()
        _camera.release_camera()

    if _shell_thread and _shell_thread.is_alive():
        _shell_thread.join(timeout=2.0)

    if _shell_proc:
        try:
            _shell_proc.terminate()
            _shell_proc.wait(timeout=2.0)
        except Exception:
            pass

    for t in _all_threads:
        if t.is_alive() and t != threading.current_thread():
            try:
                t.join(timeout=1.0)
            except Exception:
                pass

    log("Cleanup complete.")
    _hardware_initialized = False


def get_stereo_camera() -> Optional[StereoCamera]:
    return _camera


def get_camera(left: bool) -> Optional[np.ndarray]:
    camera = get_stereo_camera()
    if camera is None:
        return None
    with camera.lock:
        old_show_left = camera.show_left
    camera.update_params(show_left=left)
    frame = camera.get_frame()
    camera.update_params(show_left=old_show_left)
    return frame


def get_raw_frame(left: bool = True) -> Optional[np.ndarray]:
    camera = get_stereo_camera()
    if camera is not None:
        return camera.get_rectified_frame(left)
    return None


def angle(servo: int, angle_value: int) -> bool:
    if _servo is None:
        return False
    return _servo.set_servo(servo, angle_value, smooth=True, step_delay=0.01, step_angle=2)


def get_coords_stereo(stereo_image, x: int, y: int) -> Optional[Tuple[float, float, float]]:
    _ = stereo_image
    camera = get_stereo_camera()
    if camera is None:
        return None
    try:
        coords = camera.get_real_coords(x, y)
        if coords is None:
            return None
        return coords['x'], coords['y'], coords['z']
    except Exception as e:
        log(f"get_coords_stereo error: {e}")
        return None


def emote(emotion_name: str) -> bool:
    return set_emote(emotion_name)


def get_servo_offsets() -> Dict[int, float]:
    servo = _servo
    if servo is None:
        return {}
    with servo.lock:
        return dict(servo.offsets)


def get_servo_angles() -> Dict[int, int]:
    servo = _servo
    if servo is None:
        return {}
    try:
        with servo.lock:
            return dict(servo.current_angles)
    except Exception as e:
        log(f"Error getting servo angles: {e}")
        return {}


def get_servo_angles_physical() -> Dict[int, int]:
    servo = _servo
    if servo is None:
        return {}
    try:
        with servo.lock:
            physical_angles = {}
            for ch, angle in servo.current_angles.items():
                if ch in servo.inverted_channels:
                    min_angle, max_angle, _, _ = servo.channel_configs[ch]
                    physical_angle = max_angle - (angle - min_angle)
                    physical_angles[ch] = physical_angle
                else:
                    physical_angles[ch] = angle
            return physical_angles
    except Exception as e:
        log(f"Error getting physical servo angles: {e}")
        return {}


def set_servo_physical(channel: int, physical_angle: int) -> bool:
    servo = _servo
    if servo is None:
        return False
    if channel not in servo.channel_configs:
        return False
    min_angle, max_angle, _, _ = servo.channel_configs[channel]
    physical_angle = max(min_angle, min(max_angle, physical_angle))
    if channel in servo.inverted_channels:
        logical_angle = max_angle - (physical_angle - min_angle)
    else:
        logical_angle = physical_angle
    return servo.set_servo(channel, int(logical_angle), smooth=True, step_delay=0.01, step_angle=2)


def set_servo_offset(channel: int, offset: float) -> bool:
    servo = _servo
    if servo is None:
        return False
    return servo.set_offset(channel, offset)


def get_ai_agent():
    return _ai_agent


def command(text: str) -> None:
    """Send a command to the AI agent (for use in the Python tab)."""
    if _ai_agent is None:
        log("AI agent not initialized")
        return
    _ai_agent.command(text)
