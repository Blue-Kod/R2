import platform
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

# --- Configuration ---
APP_VERSION = "0.2"
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 80
HTTPS_PORT = 443
CAMERA_SOURCE = 0
CAMERA_PARAMS_FILE = "cam_params.json"
LAUNCHER_SCRIPT = "launcher.py"
APP_PASSWORD = "orangepi"

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Проверка пароля входа: пароль root ОС (Orange Pi — "orangepi") ---

def _root_shadow_hash() -> Optional[str]:
    """Хеш пароля root из /etc/shadow (None, если недоступен)."""
    try:
        with open("/etc/shadow", "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                name, rest = line.split(":", 1)
                if name == "root":
                    return rest.split(":", 1)[0]
    except Exception:
        return None
    return None


def check_root_password(password: str) -> bool:
    """Проверить пароль по хешу root из /etc/shadow.

    Пароль веб-панели совпадает с паролем пользователя root системы
    (по умолчанию на Orange Pi — "orangepi"). Если /etc/shadow недоступен
    (не-root процесс, Windows/mock), используется "orangepi" как запасной.
    """
    entry = _root_shadow_hash()
    locked = entry in ("!", "*", "!!") or (entry and entry.startswith(("!", "*")))
    if entry and not locked:
        try:
            import crypt  # Python <= 3.12 (удалён в 3.13+)
            if crypt.crypt(password, entry) == entry:
                return True
        except Exception:
            pass
        try:
            scheme = entry.split("$")[1] if entry.count("$") >= 3 else ""
            flag = {"1": "-1", "5": "-5", "6": "-6"}.get(scheme)
            if flag:
                salt = entry.split("$")[2]
                out = subprocess.run(
                    ["openssl", "passwd", flag, "-salt", salt, password],
                    capture_output=True, text=True, timeout=5)
                if out.returncode == 0 and out.stdout.strip() == entry:
                    return True
        except Exception:
            pass
    return password == APP_PASSWORD

# --- Global state ---
_camera: Optional[StereoCamera] = None
_servo: Optional[ServoController] = None
_lock: threading.Lock = threading.Lock()

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


def start_background() -> None:
    global _hardware_initialized, _all_threads

    if _hardware_initialized:
        return

    log(f"R2 v{APP_VERSION} - Starting...")

    _init_hardware()

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
    if servo is None:
        return
    if enable:
        if not servo.initialized:
            servo.reinit()
        # Сначала снимаем блок relax_all(), иначе set_servo() откажется
        # (сервы не включатся после выключения).
        servo.enable_all()
        for ch in servo.channel_configs:
            servo.set_servo(ch, servo.current_angles.get(ch, 90), smooth=False)
    else:
        # relax_all() сам попробует восстановить шину, если она «отвалилась».
        servo.relax_all()


def cleanup() -> None:
    global _hardware_initialized

    if not _hardware_initialized:
        return

    log("Starting clean shutdown...")

    try:
        from robov_core.web import _collector
        _collector.close()
    except Exception:
        pass

    if _servo is not None:
        log("Relaxing servos...")
        _servo.relax_all()

    if _camera:
        _camera.stop_continuous_capture()
        _camera.release_camera()

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
    return camera.get_frame(left=left)


def get_raw_frame(left: bool = True):
    camera = get_stereo_camera()
    if camera is not None:
        return camera.get_rectified_frame(left)
    return None


def angle(servo: int, angle_value: int) -> bool:
    if _servo is None:
        return False
    return _servo.set_servo(servo, angle_value, smooth=True, step_delay=0.01, step_angle=2)


def robot_config() -> dict:
    """Единый конфиг робота (серво + кинематика) для API и фронтенда.

    Единственная точка, где собирается конфиг для UI: значения берутся из
    servo.servo_config() и arm_kinematics.browser_config(), без дублей.
    """
    from robov_core.servo import servo_config
    return {"servo": servo_config(), "ik": arm_kinematics.browser_config()}


def get_servo_calibration() -> Dict[str, object]:
    """Текущая калибровка (offsets + инвертированные каналы)."""
    servo = _servo
    if servo is None:
        from robov_core.servo import CHANNELS, DEFAULT_OFFSETS, INVERTED_CHANNELS
        return {
            "offsets": {str(ch): float(DEFAULT_OFFSETS.get(ch, 0.0)) for ch in CHANNELS},
            "inverted": sorted(INVERTED_CHANNELS),
            "connected": False,
        }
    data = servo.calibration()
    data["connected"] = servo.is_connected()
    return data


def reinit_servo_bus() -> bool:
    """Переоткрыть I2C-шину PCA9685 (лечит «отвалившуюся» связь)."""
    servo = _servo
    if servo is None:
        return False
    return servo.reinit()


def set_servo_calibration(offsets=None, inverted=None) -> bool:
    """Применить калибровку к серво и сохранить в config.json."""
    servo = _servo
    if servo is None:
        return False
    if offsets:
        for ch_str, value in offsets.items():
            try:
                channel = int(ch_str)
                angle_offset = float(value)
            except (TypeError, ValueError):
                continue
            servo.set_offset(channel, angle_offset)
    if inverted is not None:
        wanted = set()
        for item in inverted:
            try:
                wanted.add(int(item))
            except (TypeError, ValueError):
                pass
        for channel in servo.channel_configs:
            servo.set_inverted(channel, channel in wanted)
    servo.save_calibration()
    return True


def reset_servo_calibration() -> bool:
    """Вернуть offsets/инверсию к значениям из servo.py и сохранить."""
    servo = _servo
    if servo is None:
        return False
    from robov_core.servo import INVERTED_CHANNELS
    servo.reset_offsets_to_default()
    for channel in servo.channel_configs:
        servo.set_inverted(channel, channel in INVERTED_CHANNELS)
    servo.save_calibration()
    return True


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


def get_servo_positions() -> Dict[int, int]:
    """Фактическое (сглаженное) положение каналов — для живого индикатора."""
    servo = _servo
    if servo is None:
        return {}
    try:
        return {int(ch): int(v) for ch, v in servo.positions().items()}
    except Exception as e:
        log(f"Error getting servo positions: {e}")
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
# Flask работает threaded=True, а телеоп шлёт IK на 20 Гц параллельно с
# web-API: без лока rate-limiter и _last_ik_start гоняются.
_ik_lock = threading.Lock()


def ik_detail(x: float, y: float, z: float, left: bool = False,
              start: Optional[Sequence[float]] = None) -> dict:
    """Полный результат IK для web/API в системе координат камеры.

    Стартовая поза для выбора ветки — последняя командованная поза руки
    (непрерывность при движении), пока она есть; иначе — текущие углы.
    """
    if start is None:
        with _ik_lock:
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
    with _ik_lock:
        for ch, angle in commands.items():
            prev = _last_cmd_angle.get(ch)
            phys = None
            if servo is not None:
                phys = float(servo.current_angles.get(ch, 90.0))
            if prev is None:
                # Первая команда канала после старта: едем к цели сразу
                # (плавность физического движения обеспечивает mover серво),
                # иначе dt=0 заморозил бы руку на первом движении.
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


def move_ik_detail(x: float, y: float, z: float, left: bool = False) -> dict:
    """Вычислить IK и двигать руку к ближайшей достижимой позе.

    Если цель недостижима, рука едет в ближайшую достижимую точку
    (решение есть в result["servo"]); не двигаемся только когда решения
    нет вообще (цель внутри туловища/головы).  Каждый вызов стартует
    от текущего положения (сброс rate-limiter), чтобы рука всегда
    ехала к полной цели, а не к промежуточной из предыдущего вызова.
    """
    with _ik_lock:
        _last_cmd_angle.clear()
        _last_cmd_time.clear()
    result = ik_detail(x, y, z, left)
    result["moved"] = False
    if not result["servo"]:
        return result
    limited = _rate_limit_commands(result["servo"])
    for channel, angle_value in limited.items():
        if not set_servo_command(channel, int(round(angle_value))):
            result["message"] += "; не удалось запустить движение серво"
            return result
    with _ik_lock:
        _last_ik_start[left] = arm_kinematics.theta_from_commands(limited, left)
    result["servo"] = {ch: int(round(v)) for ch, v in limited.items()}
    result["moved"] = True
    return result

