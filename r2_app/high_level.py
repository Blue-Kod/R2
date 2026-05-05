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

from camera import StereoCamera
from servo import ServoController

try:
    from eyes_display import EyeDisplay, optimize_for_arm
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
_camera_lock = threading.Lock()
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

servo_tracking_enabled = False
_tracking_thread = None
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
_supported_emotes = [os.path.splitext(f)[0] for f in os.listdir("emotions") if f.endswith(".png")]

_all_threads = []

# --- Автоматическое восстановление камеры ---
_camera_reinit_thread = None
_camera_reinit_interval = 30.0   # секунд между попытками
_camera_source = 0               # будет обновлён при инициализации
_camera_config_path = None


def log(message: str) -> None:
    print(message)


class MockStereoCamera:
    """Windows/dev fallback camera to keep web UI and APIs alive."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.img_size = (1280, 720)
        self.low_size = (320, 180)
        self.depth_enabled = True
        self.face_tracking_enabled = True
        self.tracking_mode = "person"
        self.tracking_scale_x = 50.0
        self.tracking_scale_y = 30.0
        self.tracking_offset_x = 0.0
        self.tracking_offset_y = 0.0
        self.alpha_depth = 0.3
        self.show_left = True
        self.num_disp = 5
        self.wls_enabled = False
        self.face_dx = 0.0
        self.face_dy = 0.0
        self.fps = 30.0
        self.points_3d = np.zeros((self.low_size[1], self.low_size[0], 3), dtype=np.float32)
        self._tick = 0

    def _make_frame(self) -> np.ndarray:
        frame = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        self._tick += 1
        cx = int((self._tick * 7) % self.img_size[0])
        cy = int(self.img_size[1] / 2 + np.sin(self._tick / 15.0) * 120)
        cv2.putText(frame, "SIMULATION MODE", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, "No camera / Camera init failed", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 40, (0, 200, 255), -1)
        return frame

    def get_rectified_frame(self, left=True):
        return self._make_frame()

    def get_frame(self):
        with self.lock:
            return self._make_frame()

    def get_depth_at(self, x, y):
        _ = (x, y)
        return 120.0

    def get_eye_offsets(self):
        with self.lock:
            return self.face_dx, self.face_dy

    def get_point_cloud_sample(self, step=2, max_distance_cm=1500):
        _ = max_distance_cm
        points = []
        for y in range(0, self.low_size[1], max(1, step)):
            for x in range(0, self.low_size[0], max(1, step)):
                points.append({"x": float(x), "y": float(y), "z": 120.0})
        return points

    def update_params(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if value is not None and hasattr(self, key):
                    setattr(self, key, value)


def _init_hardware():
    global _camera, _servo, _camera_source, _camera_config_path

    if platform.system() == "Windows":
        _camera = MockStereoCamera()
        log("Simulation mode enabled: mock camera and mock servo are active")
    else:
        try:
            from r2_app.config import AppConfig
            config = AppConfig()
            _camera_source = config.camera_source
            _camera_config_path = str(config.camera_config_path)
            if config.camera_config_path.exists():
                _camera = StereoCamera(str(config.camera_config_path), source=config.camera_source)
                log("Camera initialized")
            else:
                raise FileNotFoundError("Camera config not found")
        except Exception as exc:
            log(f"Camera init error: {exc}")
            _camera = MockStereoCamera()
            log("Falling back to simulation mode")
            # Сохраняем параметры на будущее
            try:
                from r2_app.config import AppConfig
                config = AppConfig()
                _camera_source = config.camera_source
                _camera_config_path = str(config.camera_config_path)
            except:
                _camera_source = 0
                _camera_config_path = "cam_params.json"

        try:
            _servo = ServoController(bus=0, address=0x40, freq=50)
            log("Servo controller initialized")
            from r2_app.config import AppConfig
            config = AppConfig()
            for channel, angle in config.default_servo_angles.items():
                if channel in _servo.channel_configs:
                    _servo.set_servo(channel, angle, smooth=False)
        except Exception as exc:
            log(f"Servo init error: {exc}")
            log("Falling back to mock servo")


def _camera_reinit_loop():
    """Фоновая нить: если камера мок, периодически пробует заменить на реальную."""
    global _camera, _camera_source, _camera_config_path
    while not _shutdown_requested:
        time.sleep(_camera_reinit_interval)
        if not isinstance(_camera, MockStereoCamera):
            # Уже реальная камера — выходим из цикла
            break
        log("[CAM] Attempting to switch to real camera...")
        try:
            from camera import StereoCamera as RealCamera
            if not _camera_config_path or not os.path.exists(_camera_config_path):
                log("[CAM] No config file for real camera")
                continue
            new_cam = RealCamera(_camera_config_path, source=_camera_source)
            # Если успешно, подменяем
            with _camera_lock:
                old_cam = _camera
                _camera = new_cam
            if old_cam and hasattr(old_cam, 'stop'):
                old_cam.stop()
            log("[CAM] Successfully switched to real camera")
            break
        except Exception as e:
            log(f"[CAM] Real camera still unavailable: {e}")


def reinit_camera():
    """Ручной запуск попытки восстановления камеры (можно вызвать из API)."""
    global _camera
    if isinstance(_camera, MockStereoCamera):
        log("[CAM] Manual reinit triggered")
        # Выполняем попытку немедленно в этом же потоке
        try:
            from camera import StereoCamera as RealCamera
            if not _camera_config_path or not os.path.exists(_camera_config_path):
                log("[CAM] No config file")
                return False
            new_cam = RealCamera(_camera_config_path, source=_camera_source)
            with _camera_lock:
                old_cam = _camera
                _camera = new_cam
            if old_cam and hasattr(old_cam, 'stop'):
                old_cam.stop()
            log("[CAM] Manual reinit successful")
            return True
        except Exception as e:
            log(f"[CAM] Manual reinit failed: {e}")
            return False
    else:
        log("[CAM] Already using real camera")
        return True


def _tracking_loop():
    global _camera, _servo, servo_tracking_enabled, _shutdown_requested

    default_neck = 90
    default_tilt = 90
    max_neck_delta = 30
    max_tilt_delta = 15
    last_neck = default_neck
    last_tilt = default_tilt
    last_target_time = time.time()
    target_lost = True

    while not _shutdown_requested:
        if servo_tracking_enabled and _camera and _servo:
            dx, dy = _camera.get_eye_offsets()
            scale_x = _camera.tracking_scale_x or 1.0
            scale_y = _camera.tracking_scale_y or 1.0

            target_present = (dx != 0.0 or dy != 0.0)
            if target_present:
                target_lost = False
                last_target_time = time.time()

                neck_angle = default_neck + (dx / scale_x) * max_neck_delta
                neck_angle = max(default_neck - max_neck_delta, min(default_neck + max_neck_delta, neck_angle))
                tilt_angle = default_tilt + (dy / scale_y) * max_tilt_delta
                tilt_angle = max(default_tilt - max_tilt_delta, min(default_tilt + max_tilt_delta, tilt_angle))

                if abs(neck_angle - last_neck) > 1:
                    _servo.set_servo(0, int(round(neck_angle)), smooth=True, step_delay=0.01, step_angle=2)
                    last_neck = neck_angle
                if abs(tilt_angle - last_tilt) > 1:
                    _servo.set_servo(3, int(round(tilt_angle)), smooth=True, step_delay=0.01, step_angle=2)
                    last_tilt = tilt_angle
            else:
                if not target_lost:
                    target_lost = True
                    last_target_time = time.time()
                    log(f"Target lost, waiting before returning to center")
                elif (time.time() - last_target_time) >= 5:
                    if abs(last_neck - default_neck) > 1:
                        _servo.set_servo(0, default_neck, smooth=True, step_delay=0.01, step_angle=2)
                        last_neck = default_neck
                    if abs(last_tilt - default_tilt) > 1:
                        _servo.set_servo(3, default_tilt, smooth=True, step_delay=0.01, step_angle=2)
                        last_tilt = default_tilt
        else:
            target_lost = True
            last_target_time = time.time()

        time.sleep(0.05)


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
    global _hardware_initialized, _tracking_thread, _all_threads, _camera_reinit_thread

    if _hardware_initialized:
        return

    sys.stdout = _stdout_capture
    sys.stderr = _stderr_capture

    log("R2 v2.0 - Starting...")

    _init_hardware()

    # Запускаем фоновую проверку камеры (если сейчас мок)
    if isinstance(_camera, MockStereoCamera):
        _camera_reinit_thread = threading.Thread(target=_camera_reinit_loop, daemon=True, name="r2-camera-reinit")
        _camera_reinit_thread.start()
        _all_threads.append(_camera_reinit_thread)

    _tracking_thread = threading.Thread(target=_tracking_loop, daemon=True, name="r2-tracking-thread")
    _tracking_thread.start()
    _all_threads.append(_tracking_thread)

    shell_start()
    if _shell_thread:
        _all_threads.append(_shell_thread)

    from r2_app.web import create_app
    from r2_app.config import AppConfig
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

    if _tracking_thread and _tracking_thread.is_alive():
        _tracking_thread.join(timeout=2.0)
    if _shell_thread and _shell_thread.is_alive():
        _shell_thread.join(timeout=2.0)
    if _camera_reinit_thread and _camera_reinit_thread.is_alive():
        _camera_reinit_thread.join(timeout=1.0)

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


def build_point_cloud(stereo_image, camera_image, step: int = 2):
    _ = stereo_image, camera_image
    camera = get_stereo_camera()
    if camera is None:
        return []
    return camera.get_point_cloud_sample(step=step)


def set_servo_tracking(enabled: bool):
    global servo_tracking_enabled
    servo_tracking_enabled = bool(enabled)


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