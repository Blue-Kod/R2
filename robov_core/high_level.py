import contextlib
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

import cv2
import numpy as np

from robov_core.camera import StereoCamera
from robov_core.servo import REST_POSE, ServoController
from robov_core.stereo_depth import StereoDepth, Vector3, split_stereo
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

# Режим симуляции: железа нет — фейк-камера с надписью SIMULATION, серво
# только в память, порты 8080/8443 (80/443 не открыть без root),
# выключение/перезагрузка/обновление из панели отключены.
SIMULATION: bool = False

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
        aplay = subprocess.Popen(
            ["aplay", "-D", "plughw:1,0", "-t", "raw",
             "-f", "S16_LE", "-r", "22050", "-c", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        aplay.stdin.write(raw_audio)
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
            _camera = StereoCamera(str(cam_path), source=CAMERA_SOURCE,
                                   simulation=SIMULATION)
            if _camera.initialize_camera():
                _camera.start_continuous_capture()
                log(f"Camera initialized on {platform.system()}")
            else:
                log("Camera capture failed - continuing with mock camera")
        except Exception as exc:
            log(f"Camera init error: {exc} - continuing with mock camera")
    else:
        log("Camera config not found - continuing with mock camera")

    if SIMULATION:
        try:
            _servo = ServoController(simulation=True)
            log("Servo controller: SIMULATION (углы в память, без I2C/PWM)")
            for channel, angle in _servo.current_angles.items():
                if channel in _servo.channel_configs:
                    _servo.set_servo(channel, angle, smooth=False)
        except Exception as exc:
            log(f"Servo init error: {exc}")
    elif platform.system() == "Windows":
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


def start_background(simulation: bool = False) -> None:
    global _hardware_initialized, _all_threads, SIMULATION, HTTP_PORT, HTTPS_PORT

    if _hardware_initialized:
        return

    SIMULATION = simulation
    if simulation:
        # 80/443 непривилегированному процессу не открыть (не-root машина).
        HTTP_PORT, HTTPS_PORT = 8080, 8443

    log(f"R2 v{APP_VERSION} - Starting..." + (" [SIMULATION]" if simulation else ""))

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

    if simulation:
        log(f"SIMULATION: панель http://<ip>:{HTTP_PORT} -> https://<ip>:{HTTPS_PORT}")

    # Initialize TTS early so it's ready when needed
    try:
        _init_tts()
    except Exception as e:
        log(f"TTS init error: {e}")

    # WiFi QR setup — if no internet, scan for WiFi QR codes via camera
    # (в симуляции не запускаем: не трогаем сетевые настройки машины)
    from robov_core.qr_wifi import check_internet, start_wifi_setup
    if not simulation and not check_internet():
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


# --- Глубина по стереопаре -------------------------------------------------
DEPTH_CALIB = ROOT_DIR / "calib.json"

_depth_engine: Optional[StereoDepth] = None
_depth_frame: Tuple[Optional[int], Optional[np.ndarray]] = (None, None)
_depth_lock: threading.Lock = threading.Lock()

DEPTH_WAIT = 0.5     # с: ждём кадр, только если камера ещё ничего не отдала


def _stereo_depth() -> StereoDepth:
    """Движок глубины; создаётся при первом обращении из calib.json."""
    global _depth_engine
    if _depth_engine is None:
        _depth_engine = StereoDepth(DEPTH_CALIB)
    return _depth_engine


def _stereo_frame() -> Optional[np.ndarray]:
    """Последний стерео-кадр (серый 2560x720), декодированный с кэшем по номеру.

    Свежий кадр не ждём: ffmpeg публикует его каждый кадр, самый старый здесь
    отстаёт на 1/13 с. Подписка на стерео нужна только OpenCV-бэкенду, где
    кадр кодируется лишь при наличии потребителя.
    """
    global _depth_frame
    cam = _camera
    if cam is None:
        return None
    cam.add_consumer(stereo=True)
    try:
        jpeg, seq = cam.get_stereo_jpeg()
        deadline = time.monotonic() + DEPTH_WAIT
        while jpeg is None and time.monotonic() < deadline:
            time.sleep(0.01)
            jpeg, seq = cam.get_stereo_jpeg()
    finally:
        cam.remove_consumer(stereo=True)
    if jpeg is None:
        return None
    with _depth_lock:
        if _depth_frame[0] == seq and _depth_frame[1] is not None:
            return _depth_frame[1]
    # серый: глубине цвет не нужен, а декод 2560x720 в серый в 8 раз быстрее
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    with _depth_lock:
        _depth_frame = (seq, img)
    return img


def _depth_sample(x: float, y: float):
    """(движок, левый, правый, px, py) — доли кадра 0..1 переводятся в пиксели."""
    img = _stereo_frame()
    if img is None:
        return None, None, None, None, None
    left, right = split_stereo(img)
    if right is None:
        return None, None, None, None, None
    eng = _stereo_depth()
    w, h = eng.size or (left.shape[1], left.shape[0])
    return eng, left, right, x * w, y * h


def _shown_eye() -> str:
    return "left" if _camera is not None and _camera.show_left else "right"


def depth(x: float, y: float, window: int = 2) -> float:
    """Расстояние до точки кадра, метры. NaN — если данных нет.

    x, y — доли кадра от 0 до 1 (x вправо, y вниз, как пиксели). Считается
    локально вокруг точки (патч 33x33), поэтому запрос стоит ~20 мс. Глубина —
    медиана окна, окно расширяется, если в нём нет валидных пикселей.

    Это глубина по оси ВЫПРЯМЛЕННОГО кадра (метры, от камеры). coords() даёт
    точку в системе корпуса: у точки далеко от центра её z отличается на
    несколько процентов — это поворот ректификации, а не ошибка; для IK
    используйте coords().
    """
    eng, left, right, px, py = _depth_sample(x, y)
    if left is None:
        return float("nan")
    return eng.depth_point(left, right, px, py, eye=_shown_eye(), window=window)


def coords(x: float, y: float, window: int = 2) -> Vector3:
    """3D-точка кадра (x, y от 0 до 1) в системе КОРПУСА: X вправо, Y вверх,
    Z вперёд, МЕТРЫ.

    Сначала точка считается в системе камеры, затем поворачивается на текущие
    углы шеи (канал 0) и наклона (канал 3) — те команды, что отправлены сервам.
    Начало — центр корпуса, ноль по глубине — ось рук. Единицы — МЕТРЫ (как у
    depth()). z здесь больше depth() на путь от объектива до центра головы
    (HEAD_LENS_Z = 45 мм) и на поворот, если шея/наклон не в нуле. В IK
    (ik_detail/move_ik_detail) внутри переводятся в миллиметры — там так
    заданы звенья.
    """
    eng, left, right, px, py = _depth_sample(x, y)
    if left is None:
        return Vector3(float("nan"), float("nan"), float("nan"))
    p = eng.point_at(left, right, px, py, eye=_shown_eye(), window=window)
    if not np.isfinite(p.x):
        return p
    # левый глаз -> середина стереобазы (там ось головы), мм для кинематики
    pc = ((p.x + eng.baseline / 2.0) * 1000.0, p.y * 1000.0, p.z * 1000.0)
    neck = tilt = arm_kinematics.HEAD_REST
    servo = _servo
    if servo is not None:
        neck = float(servo.current_angles.get(arm_kinematics.NECK_CHANNEL, neck))
        tilt = float(servo.current_angles.get(arm_kinematics.TILT_CHANNEL, tilt))
    q = arm_kinematics.camera_to_body(pc, neck, tilt)   # мм
    return Vector3(float(q[0]) / 1000.0, float(q[1]) / 1000.0,
                   float(q[2]) / 1000.0)


# --- Захват и движение в точку кадра ---------------------------------------
GRIP_CLOSE = 180            # сомкнут (как GRIP_CLOSE в templates/webxr.html)
GRIP_OPEN = 0
GRIP_CHANNEL = {False: 8, True: 9}    # 8 — правый, 9 — левый
GRIP_SETTLE = 0.3           # с: даём серве захвата доехать/схватить
MOVE_TIMEOUT = 10.0         # с: дольше этого считаем, что рука не доехала
APPROACH_UP = 0.20          # м: над объектом, откуда начинается заход
UP_MARGIN = 0.02            # м: если объект не ниже кисти на этот запас — сначала вверх
ARM_CHANNELS_POSE = (1, 2, 4, 5, 6, 7)   # каналы рук (без захватов и колёс)
# Высокоуровневые команды движения выполняются строго по одной: второй вызов
# ждёт, пока первый доведёт руку до конца (иначе IK-цели перебивают друг друга).
_motion_lock = threading.RLock()
MOVE_SPEED_DEG = 60.0       # град/с: переезды руки в задачах захвата
DESCENT_SPEED_DEG = 30.0    # град/с: заход к объекту сверху — вдвое медленнее


def _wait_motion(timeout: float = MOVE_TIMEOUT) -> bool:
    servo = _servo
    return False if servo is None else servo.wait_idle(timeout)


def grab(left: bool = False) -> bool:
    """Сжать захват (180°) соответствующей руки. True — команда ушла."""
    return set_servo_command(GRIP_CHANNEL[bool(left)], GRIP_CLOSE)


def release(left: bool = False) -> bool:
    """Разжать захват (0°) соответствующей руки. True — команда ушла."""
    return set_servo_command(GRIP_CHANNEL[bool(left)], GRIP_OPEN)


# Именованные позы меняют ТОЛЬКО руки (1, 2, 4, 5, 6, 7). Голова (0, 3),
# захваты (8, 9) и колёса (10, 11) позой не трогаются: захваты — grab/release,
# голову — set_servo_command.
POSES: Dict[str, Dict[int, int]] = {
    # руки вниз, локти выпрямлены: нулевая поза модели (servo.REST_POSE)
    "rest": {ch: int(REST_POSE[ch]) for ch in (1, 2, 4, 5, 6, 7)},
    # «руки сверху» с живого робота: плечи подняты, локти согнуты
    "ready_to_grab": {
        1: 100, 2: 100,        # плечи
        4: 220, 5: 220,        # поворот плеч
        6: 50, 7: 50,          # локти
    },
}


POSE_DEFAULT = "rest"
# Порядок групп каналов для каждой позы:
#   rest          — сначала плечи (1,2 и 4,5), потом локти;
#   ready_to_grab — сначала локти, потом плечи.
# Внутри группы каналы идут одновременно, между группами ждём доводки.
POSE_ORDER = {
    "rest": ((1, 2), (4, 5), (6, 7)),
    "ready_to_grab": ((6, 7), (4, 5), (1, 2)),
}


def pose(name: str = POSE_DEFAULT) -> dict:
    """Поставить робота в именованную позу и дождаться приезда.

    "rest" (по умолчанию) — руки вниз, локти выпрямлены (servo.REST_POSE,
    нулевая поза модели); "ready_to_grab" — руки сверху (плечи подняты,
    локти согнуты) — значения сняты с робота. Меняются только каналы рук:
    голова, захваты и колёса остаются как были.
    Захваты не трогаются: их ставят grab/release.
    """
    commands = POSES.get(name)
    if commands is None:
        raise ValueError(f"нет позы {name!r}; есть: {', '.join(POSES)}")
    with _motion_lock:
        return _pose_locked(name, commands)


def _pose_locked(name: str, commands: Dict[int, int]) -> dict:
    sent = {}
    arrived = True
    with _arm_slow(MOVE_SPEED_DEG):          # поза ставится медленно, без рывка
        for group in POSE_ORDER.get(name, ()):
            for channel in group:
                if channel in commands and set_servo_command(channel, int(commands[channel])):
                    sent[channel] = int(commands[channel])
            arrived = _wait_motion() and arrived   # доводим группу до конца
    return {"ok": bool(sent), "pose": name, "servo": sent, "arrived": arrived}


def _move_to(point: Vector3, left: bool = False) -> dict:
    """Команда IK в точку (coords в метрах, IK в миллиметрах)."""
    return move_ik_detail(point.x * 1000.0, point.y * 1000.0,
                          point.z * 1000.0, left=left)


@contextlib.contextmanager
def _arm_slow(deg_per_s: float = MOVE_SPEED_DEG):
    """Медленное движение руки: ограничиваем скорость mover'а.

    Раньше «медленно» делалось шагами по 12 мм каждые 60 мс — mover (300 °/с)
    успевал разгоняться и тормозить на каждом шаге, отсюда дёрганье и паузы.
    Здесь одна команда, а mover ведёт её трапецией на заданной скорости.
    """
    servo = _servo
    if servo is None:
        yield
        return
    servo.set_speed_limit(ARM_CHANNELS_POSE, deg_per_s)
    try:
        yield
    finally:
        servo.set_speed_limit(ARM_CHANNELS_POSE, None)


def _current_point(left: bool = False) -> Optional[Vector3]:
    """Где кисть по последним командам, м в системе IK (для «сначала вверх»)."""
    servo = _servo
    if servo is None:
        return None
    channels = arm_kinematics.ARM_CHANNELS["left" if left else "right"]
    try:
        theta = arm_kinematics.theta_from_commands(
            {ch: float(servo.current_angles.get(ch, 90)) for ch in channels.values()},
            left=left)
        ee = arm_kinematics.fk(theta, left)["EE"]
        return Vector3(float(ee[0]) / 1000.0, float(ee[1]) / 1000.0,
                       -float(ee[2]) / 1000.0)
    except Exception:                                   # noqa: BLE001
        return None


def _lift_if_needed(target: Vector3, left: bool = False) -> None:
    """Поднять руку строго вверх, если объект на уровне кисти или выше.

    Заход сбоку на такой высоте тащит кисть через стол и корпус, поэтому сначала
    уходим вверх (не смещаясь к объекту), и только потом едем к нему обычной
    программой. Робот перед такими функциями стоит в ready_to_grab, кисть уже
    поднята — оттуда подъём короткий.
    """
    current = _current_point(left)
    if current is None:
        return
    above_y = target.y + APPROACH_UP
    if above_y > current.y - UP_MARGIN:      # объект рядом или выше кисти
        _move_to(Vector3(current.x, above_y, current.z), left)
        _wait_motion()


def _reachable(point: Vector3, left: bool = False):
    """Ближайшая ДОСТИЖИМАЯ точка (м) и результат ik_detail.

    Нужна потому, что IK умеет возвращать ближайшую позу, если цель дальше
    вылета. Если строить «выше на 20 см» от недостижимой цели, обе фазы
    схлопываются в один и тот же упор — рука идёт сразу к объекту.
    """
    result = ik_detail(point.x * 1000.0, point.y * 1000.0,
                       point.z * 1000.0, left=left)
    ee = result.get("ee")
    if ee is None:
        return None, result
    return Vector3(float(ee[0]) / 1000.0, float(ee[1]) / 1000.0,
                   -float(ee[2]) / 1000.0), result


def _approach(point: Vector3, left: bool = False) -> dict:
    """Заход сверху: переезд на APPROACH_UP выше точки, затем медленный спуск.

    point — уже достижимая точка (см. _reachable).
    """
    above = Vector3(point.x, point.y + APPROACH_UP, point.z)
    with _arm_slow(MOVE_SPEED_DEG):
        result = _move_to(above, left)
        _wait_motion()
    with _arm_slow(DESCENT_SPEED_DEG):          # заход сверху — медленно
        result = _move_to(point, left)
        result["arrived"] = _wait_motion()
    return result


def hand_go_to_point(x: float, y: float, left: bool = False) -> dict:
    """Вести руку в точку кадра (x, y от 0 до 1) и дождаться приезда.

    Точка берётся через coords() — та же система, что у ik_detail. Если
    глубины в этом пикселе нет, рука не двигается.
    """
    point = coords(x, y)
    if not np.isfinite(point.x):
        return {"ok": False, "moved": False, "arrived": False,
                "message": "нет глубины в этой точке кадра",
                "point": point}
    with _motion_lock, _arm_slow(MOVE_SPEED_DEG):
        result = _move_to(point, left)
        result["point"] = point
        result["arrived"] = bool(result.get("moved")) and _wait_motion()
    return result


def pick_point(x: float, y: float, left: bool = False) -> dict:
    """Взять объект: раскрыть захват -> зайти на 20 см выше -> медленно спуститься
    -> сжать захват. Сжатие — в самом конце, после всех движений.
    """
    return _point_action(x, y, GRIP_OPEN, GRIP_CLOSE, left=left)


def place_point(x: float, y: float, left: bool = False) -> dict:
    """Положить: сжать захват -> зайти выше -> медленно спуститься -> разжать
    -> подняться на 20 см.
    """
    return _point_action(x, y, GRIP_CLOSE, GRIP_OPEN, left=left, lift_after=True)


def _point_action(x: float, y: float, grip_before: int, grip_after: int,
                  left: bool = False, lift_after: bool = False) -> dict:
    with _motion_lock:
        return _point_action_locked(x, y, grip_before, grip_after, left, lift_after)


def _point_action_locked(x: float, y: float, grip_before: int, grip_after: int,
                         left: bool, lift_after: bool) -> dict:
    channel = GRIP_CHANNEL[bool(left)]
    set_servo_command(channel, grip_before)   # сначала клешня, потом движение
    time.sleep(GRIP_SETTLE)                   # и кадр для цели — уже без неё
    point = coords(x, y)
    if not np.isfinite(point.x):
        return {"ok": False, "moved": False, "arrived": False,
                "message": "нет глубины в этой точке кадра",
                "point": point}
    base, ik = _reachable(point, left)
    if base is None:
        return {"ok": False, "moved": False, "arrived": False,
                "message": ik.get("message") or "IK не нашёл решения",
                "point": point}
    with _arm_slow(MOVE_SPEED_DEG):
        _lift_if_needed(base, left)      # объект выше/на уровне кисти — сначала вверх
    result = _approach(base, left)
    # Захват переключаем после всех движений и независимо от того, дошла ли рука
    # до цели: оставить клешню в прежнем состоянии хуже, чем сжать/разжать.
    set_servo_command(channel, grip_after)
    time.sleep(GRIP_SETTLE)
    if lift_after:
        with _arm_slow(MOVE_SPEED_DEG):
            _move_to(Vector3(base.x, base.y + APPROACH_UP, base.z), left)
            result["arrived"] = _wait_motion()
    result["point"] = point
    result["reachable"] = base
    result["grip"] = grip_after
    return result


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

