# high_level.py
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Optional

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

# Global state
_camera = None
_servo = None
_display = None
_eye_api = None
_lock = threading.Lock()
_servo_lock = threading.Lock()
_display_lock = threading.Lock()

_logs_buffer = deque(maxlen=500)

class StdoutCapture:
    def __init__(self):
        self._original_stdout = sys.stdout
        self._lock = threading.Lock()

    def write(self, message):
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        self._original_stdout.write(message)
        self._original_stdout.flush()
        if message.strip():
            with self._lock:
                for line in message.strip().split('\n'):
                    if line.strip():
                        _logs_buffer.append(line)

    def flush(self):
        self._original_stdout.flush()

_stdout_capture = StdoutCapture()
_stderr_capture = StdoutCapture()

_hardware_initialized = False
_shutdown_requested = False

_shell_proc = None
_shell_buffer = deque(maxlen=2000)
_shell_running = False
_shell_lock = threading.Lock()
_shell_thread = None

_current_emote = "normal"
_eyes_x = 0.0
_eyes_y = 0.0
_emote_lock = threading.Lock()
_emotions_dir = os.path.join(os.path.dirname(__file__), "emotions")
_supported_emotes = [os.path.splitext(f)[0] for f in os.listdir(_emotions_dir) if f.endswith(".png")]

_all_threads = []

def log(message: str) -> None:
    print(message)

def _init_hardware():
    global _camera, _servo
    from robov_core.config import AppConfig
    config = AppConfig()
    if config.camera_config_path.exists():
        _camera = StereoCamera(str(config.camera_config_path), source=config.camera_source)
        if _camera.initialize_camera():
            log(f"Camera initialized on {platform.system()}")
        else:
            log(f"Camera capture failed on {platform.system()} - continuing with mock camera")
            # Don't raise error, continue with failed camera that will show "CAMERA ERROR"
    else:
        log("Camera config not found - continuing with mock camera")
        # Don't raise error, continue without camera

    # Initialize servo controller (platform-specific)
    if platform.system() == "Windows":
        log("Mock servo mode on Windows")
    else:
        try:
            _servo = ServoController(bus=0, address=0x40, freq=50)
            log("Servo controller initialized")
            from robov_core.config import AppConfig
            config = AppConfig()
            for channel, angle in config.default_servo_angles.items():
                if channel in _servo.channel_configs:
                    _servo.set_servo(channel, angle, smooth=False)
        except Exception as exc:
            log(f"Servo init error: {exc}")
            log("Falling back to mock servo")


def _shell_reader():
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


def shell_start():
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
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Failed to push emote to display: {e}")

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
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Failed to push eyes position to display: {e}")


def get_eyes_position():
    with _emote_lock:
        return _eyes_x, _eyes_y


def supported_emotes():
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


def get_logs(n: int = 500) -> list:
    with _lock:
        return list(_logs_buffer)[-n:]


def health_snapshot() -> dict:
    return {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "temp": cpu_temp(),
    }


def _display_worker():
    global _display, _eye_api

    if not _HAS_DISPLAY:
        print("[!] eyes_display.py not available, skipping pywebview display")
        return

    if optimize_for_arm:
        optimize_for_arm()

    with _display_lock:
        _display = EyeDisplay()

    _eye_api = _display.api

    _display.start()


def start_background():
    global _hardware_initialized, _all_threads

    if _hardware_initialized:
        return

    sys.stdout = _stdout_capture
    sys.stderr = _stderr_capture

    log("R2 v2.0 - Starting...")

    _init_hardware()

    shell_start()
    if _shell_thread:
        _all_threads.append(_shell_thread)

    from robov_core.web import create_app
    from robov_core.config import AppConfig
    config = AppConfig()
    app = create_app()
    web_thread = threading.Thread(
        target=lambda: app.run(host=config.host, port=config.http_port, debug=False, threaded=True, use_reloader=False),
        daemon=True,
        name="r2-web-thread"
    )
    web_thread.start()
    _all_threads.append(web_thread)

    if _HAS_DISPLAY:
        display_thread = threading.Thread(target=_display_worker, daemon=True, name="r2-display-thread")
        display_thread.start()
        _all_threads.append(display_thread)

    _hardware_initialized = True


def cleanup():
    global _shutdown_requested, _hardware_initialized

    if not _hardware_initialized:
        return

    log("Starting clean shutdown...")
    _shutdown_requested = True
    _shell_running = False

    if _shell_thread and _shell_thread.is_alive():
        _shell_thread.join(timeout=2.0)

    if _shell_proc:
        try:
            _shell_proc.terminate()
            _shell_proc.wait(timeout=2.0)
        except:
            pass

    for t in _all_threads:
        if t.is_alive() and t != threading.current_thread():
            try:
                t.join(timeout=1.0)
            except:
                pass

    log("Cleanup complete.")
    _hardware_initialized = False


def get_stereo_camera():
    return _camera


def get_camera(left: bool):
    camera = get_stereo_camera()
    if camera is None:
        return None
    with camera.lock:
        old_show_left = camera.show_left
    camera.update_params(show_left=left)
    frame = camera.get_frame()
    camera.update_params(show_left=old_show_left)
    return frame


def get_raw_frame(left=True):
    camera = get_stereo_camera()
    if camera is not None:
        return camera.get_rectified_frame(left)
    return None


def angle(servo: int, angle_value: int):
    if _servo is None:
        return False
    return _servo.set_servo(servo, angle_value, smooth=True, step_delay=0.01, step_angle=2)


def get_coords_stereo(stereo_image, x: int, y: int):
    _ = stereo_image
    camera = get_stereo_camera()
    if camera is None:
        return None
    with camera.lock:
        if camera.points_3d is None:
            return None
        scale_x = camera.low_size[0] / camera.img_size[0]
        scale_y = camera.low_size[1] / camera.img_size[1]
        lx = int(x * scale_x)
        ly = int(y * scale_y)
        if lx < 0 or lx >= camera.low_size[0] or ly < 0 or ly >= camera.low_size[1]:
            return None
        px, py, pz = camera.points_3d[ly, lx]
        if pz <= 0:
            return None
        return float(px / 10.0), float(py / 10.0), float(pz / 10.0)




def emote(emotion_name: str):
    return set_emote(emotion_name)


def set_eyes_position(x, y):
    return set_eyes_position(x, y)

def get_servo_offsets():
    """Возвращает словарь {channel: offset} текущих оффсетов."""
    servo = _servo
    if servo is None:
        return {}
    with servo.lock:
        return dict(servo.offsets)

def get_servo_angles():
    """Возвращает словарь {channel: angle} текущих углов сервоприводов."""
    servo = _servo
    if servo is None:
        return {}
    angles = {}
    try:
        # Получаем текущие углы из конфигурации сервоприводов
        if hasattr(servo, 'channel_configs'):
            for channel in servo.channel_configs:
                # Здесь можно добавить логику для получения реальных углов
                # Пока используем значения по умолчанию из конфигурации
                angles[channel] = servo.channel_configs[channel][0]  # min_angle как заглушка
    except Exception as e:
        print(f"Error getting servo angles: {e}")
    return angles

def set_servo_offset(channel: int, offset: float):
    """Устанавливает оффсет сервоканала."""
    servo = _servo
    if servo is None:
        return False
    return servo.set_offset(channel, offset)

def start():
    start_background()
    while True:
        time.sleep(0.01)