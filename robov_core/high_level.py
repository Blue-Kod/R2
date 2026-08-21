import os
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List, Sequence, Tuple

import psutil

from robov_core.camera import StereoCamera
from robov_core.servo import ServoController
from robov_core import arm_kinematics

_HAS_DISPLAY = False
EyeDisplay = None
optimize_for_arm = None

# --- Configuration ---
APP_VERSION = "0.2"
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 80
HTTPS_PORT = 443
CAMERA_SOURCE = 0
CAMERA_PARAMS_FILE = "cam_params.json"
LAUNCHER_SCRIPT = "launcher.py"
EYES_SCALE_FACTOR = 1.3
APP_PASSWORD = "admin."

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
_supported_emotes: List[str] = []
if os.path.isdir(_emotions_dir):
    _supported_emotes = [os.path.splitext(f)[0] for f in os.listdir(_emotions_dir) if f.endswith(".png")]

_display_thread: Optional[threading.Thread] = None
_all_threads: List[threading.Thread] = []

_tts_ready: bool = False
_tts_lock: threading.Lock = threading.Lock()

_ESPEAK_VOICE = "ru"
_ESPEAK_SPEED = 90
_ESPEAK_PITCH = 40


def _init_tts():
    """Check that espeak-ng is available."""
    global _tts_ready
    try:
        subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True, timeout=5,
        )
        _tts_ready = True
        log("espeak-ng TTS ready")
    except FileNotFoundError:
        log("TTS unavailable: espeak-ng not installed")
    except Exception as e:
        log(f"TTS init error: {e}")


def speak(text: str) -> None:
    global _tts_ready
    if not _tts_ready:
        _init_tts()
    if not _tts_ready:
        return

    try:
        espeak = subprocess.Popen(
            ["espeak-ng", "-v", _ESPEAK_VOICE,
             "-s", str(_ESPEAK_SPEED), "-p", str(_ESPEAK_PITCH),
             "--stdout", text],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        raw_audio = espeak.stdout.read()
        espeak.wait()
        import audioop
        amplified = audioop.mul(raw_audio, 2, 1.0)
        aplay = subprocess.Popen(
            ["aplay", "-D", "plughw:1,0", "-t", "raw",
             "-f", "S16_LE", "-r", "22050", "-c", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        aplay.stdin.write(amplified)
        aplay.stdin.close()
        aplay.wait()
    except Exception as e:
        log(f"TTS error: {e}")


def _strip_speech(text: str) -> str:
    """Remove all markdown, HTML, code blocks, and formatting from TTS text."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`\n]+`", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"---+", "", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"\n{2,}", " ", text)
    return text.strip()


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
            # Поза по умолчанию — из servo.py (current_angles)
            for channel, angle in _servo.current_angles.items():
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
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    web_thread = threading.Thread(
        target=lambda: app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True, use_reloader=False),
        daemon=True,
        name="r2-web-thread"
    )
    web_thread.start()
    _all_threads.append(web_thread)

    try:
        from robov_core.tls import get_cert_paths
        cert_file, key_file = get_cert_paths()
        https_thread = threading.Thread(
            target=lambda: app.run(host=HTTP_HOST, port=HTTPS_PORT, debug=False,
                                   threaded=True, use_reloader=False,
                                   ssl_context=(str(cert_file), str(key_file))),
            daemon=True,
            name="r2-https-thread"
        )
        https_thread.start()
        _all_threads.append(https_thread)
        log(f"HTTPS: https://<robot-ip>:{HTTPS_PORT}/webxr (self-signed, подтвердите в браузере шлема)")
    except Exception as e:
        log(f"HTTPS disabled: {e}")

    if _HAS_DISPLAY:
        global _display_thread
        _display_thread = threading.Thread(target=_display_worker, daemon=True, name="r2-display-thread")
        _display_thread.start()
        _all_threads.append(_display_thread)

    # Initialize TTS early so it's ready when needed
    try:
        _init_tts()
    except Exception as e:
        log(f"TTS init error: {e}")

    # WiFi QR setup — if no internet, scan for WiFi QR codes via camera
    from robov_core.qr_wifi import check_internet, start_wifi_setup
    if not check_internet():
        wifi_thread = threading.Thread(
            target=start_wifi_setup,
            args=(speak, log),
            daemon=True,
            name="r2-wifi-setup"
        )
        wifi_thread.start()

    _hardware_initialized = True


def servo_toggle(enable: bool) -> None:
    servo = _servo
    if servo is None or not servo.initialized:
        return
    if enable:
        # Сначала снимаем блок relax_all(), иначе set_servo() откажется
        # (сервы не включатся после выключения).
        servo.enable_all()
        for ch in servo.channel_configs:
            servo.set_servo(ch, servo.current_angles.get(ch, 90), smooth=False)
    else:
        servo.relax_all()


def cleanup() -> None:
    global _shutdown_requested, _hardware_initialized, _shell_running

    if not _hardware_initialized:
        return

    log("Starting clean shutdown...")
    _shutdown_requested = True
    _shell_running = False

    stop_display()

    if _servo is not None:
        log("Relaxing servos...")
        _servo.relax_all()

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


def get_camera(left: bool) -> Optional:
    camera = get_stereo_camera()
    if camera is None:
        return None
    with camera.lock:
        old_show_left = camera.show_left
    camera.update_params(show_left=left)
    frame = camera.get_frame()
    camera.update_params(show_left=old_show_left)
    return frame


def get_raw_frame(left: bool = True):
    camera = get_stereo_camera()
    if camera is not None:
        return camera.get_rectified_frame(left)
    return None


def angle(servo: int, angle_value: int) -> bool:
    if _servo is None:
        return False
    return _servo.set_servo(servo, angle_value, smooth=True, step_delay=0.01, step_angle=2)


def goto(target) -> bool:
    log(f"goto() stub called — target={target}")
    return False


def grab(target) -> bool:
    log(f"grab() stub called — target={target}")
    return False


def move_arm_to(target, left: bool = False) -> bool:
    log(f"move_arm_to() stub called — target={target}, left={left}")
    return False


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


def get_servo_limits() -> Dict[int, List[int]]:
    """Логические диапазоны команд [min, max] из servo.py."""
    servo = _servo
    if servo is None:
        return {}
    try:
        with servo.lock:
            return {ch: list(servo.command_limits.get(ch, cfg[:2]))
                    for ch, cfg in servo.channel_configs.items()}
    except Exception as e:
        log(f"Error getting servo limits: {e}")
        return {}


def set_servo_command(channel: int, angle: int) -> bool:
    """Установить угол серво (логический, как в servo.py/high_level.py).

    Инверсию правых каналов (INVERTED_CHANNELS) применяет
    ServoController.set_servo внутри. smooth=True — плавное движение
    с разгоном/торможением через per-channel mover-поток.
    """
    servo = _servo
    if servo is None:
        return False
    if channel not in servo.channel_configs:
        return False
    min_angle, max_angle = servo.command_limits.get(
        channel, servo.channel_configs[channel][:2])
    angle = int(max(min_angle, min(max_angle, angle)))
    return servo.set_servo(channel, angle, smooth=True)


# Последний командованный theta на сторону: старт для непрерывности ветки IK
# (а не физические углы, которые отстают от команд на ходу).
_last_ik_start: Dict[bool, Optional[Tuple[float, float, float]]] = {
    False: None, True: None}
# Rate-limit команд рук: предельная скорость изменения угла на ось, град/с.
# Ограничивает прирост команды за вызов, чтобы смена ветки IK происходила
# плавным поворотом, а не мгновенным «перелётом».
MOVE_RATE_DEG_S = 150.0
_last_cmd_angle: Dict[int, float] = {}
_last_cmd_time: Dict[int, float] = {}


def ik_detail(x: float, y: float, z: float, left: bool = False,
              start: Optional[Sequence[float]] = None) -> dict:
    """Полный результат IK для web/API в системе координат камеры.

    Стартовая поза для выбора ветки — последняя командованная поза руки
    (непрерывность при движении), пока она есть; иначе — текущие углы.
    """
    if start is None:
        start = _last_ik_start[left]
    if start is None:
        servo = _servo
        if servo is not None:
            channels = arm_kinematics.ARM_CHANNELS["left" if left else "right"]
            start = arm_kinematics.theta_from_commands(
                {ch: servo.current_angles.get(ch, 90) for ch in channels.values()},
                left=left)
    return arm_kinematics.ik_solve(x, y, z, left=left, start=start)


def _rate_limit_commands(commands: Dict[int, float]) -> Dict[int, float]:
    """Ограничить прирост каждого канала скоростью MOVE_RATE_DEG_S.

    После паузы (>0.5 с) или расхождения с фактическим положением серво
    (ручное движение / внешняя команда) стартуем от фактического угла,
    чтобы рука не «рванула» с устаревшей точки.
    """
    now = time.monotonic()
    servo = _servo
    limited: Dict[int, float] = {}
    for ch, angle in commands.items():
        prev = _last_cmd_angle.get(ch)
        phys = None
        if servo is not None:
            phys = float(servo.current_angles.get(ch, 90.0))
        if prev is None:
            # Первая команда канала после старта: едем к цели сразу (плавность
            # физического движения обеспечивает mover серво), иначе dt=0
            # заморозил бы руку на первом движении.
            prev = phys if phys is not None else float(angle)
            value = float(angle)
        else:
            age = now - _last_cmd_time.get(ch, now)
            if phys is not None and age > 0.5 and abs(prev - phys) > 1.0:
                prev = phys
            dt = max(0.0, now - _last_cmd_time.get(ch, now))
            cap = MOVE_RATE_DEG_S * dt
            value = prev + max(-cap, min(cap, float(angle) - prev))
        limited[ch] = value
        _last_cmd_angle[ch] = value
        _last_cmd_time[ch] = now
    return limited


def _ik_tuple(result: dict, left: bool) -> Tuple[bool, float, float, float]:
    """Публичный компактный формат: ok, pan, shoulder, elbow."""
    commands = result.get("servo")
    if not commands:
        return bool(result.get("ok")), 0.0, 0.0, 0.0
    channels = arm_kinematics.ARM_CHANNELS["left" if left else "right"]
    return (
        bool(result.get("ok")),
        float(commands[channels["shoulder_z"]]),
        float(commands[channels["shoulder_x"]]),
        float(commands[channels["elbow_x"]]),
    )


def ik(x: float, y: float, z: float, left: bool = False) -> Tuple[bool, float, float, float]:
    """Вернуть (достижима_в_пределах_1см, pan, shoulder, elbow).

    Углы — логические команды серво, в порядке pan, shoulder, elbow.
    """
    return _ik_tuple(ik_detail(x, y, z, left), left)


def move_ik_detail(x: float, y: float, z: float, left: bool = False) -> dict:
    """Вычислить IK и двигать руку к ближайшей достижимой позе.

    Если цель недостижима, рука едет в ближайшую достижимую точку
    (решение есть в result["servo"]); не двигаемся только когда решения
    нет вообще (цель внутри туловища/головы). Команды дополнительно
    ограничиваются скоростью MOVE_RATE_DEG_S: смена ветки IK идёт плавным
    поворотом, а не мгновенным скачком.
    """
    result = ik_detail(x, y, z, left)
    result["moved"] = False
    if not result["servo"]:
        return result
    limited = _rate_limit_commands(result["servo"])
    for channel, angle_value in limited.items():
        if not set_servo_command(channel, int(round(angle_value))):
            result["message"] += "; не удалось запустить движение серво"
            return result
    _last_ik_start[left] = arm_kinematics.theta_from_commands(limited, left)
    result["moved"] = True
    return result


def move_to_ik(x: float, y: float, z: float, left: bool = False) -> Tuple[bool, float, float, float]:
    """Вычислить IK и переместить выбранную руку к ближайшей достижимой позе."""
    detail = ik_detail(x, y, z, left)
    if not detail["servo"]:
        return _ik_tuple(detail, left)
    for channel, angle_value in detail["servo"].items():
        set_servo_command(channel, int(round(angle_value)))
    return _ik_tuple(detail, left)


def set_servo_offset(channel: int, offset: float) -> bool:
    servo = _servo
    if servo is None:
        return False
    return servo.set_offset(channel, offset)
