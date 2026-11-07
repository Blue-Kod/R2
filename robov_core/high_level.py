import contextlib
import json
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
from robov_core import gripper_marker

arm_kinematics.load_tool_offsets()

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


def _p_to_body(p: Vector3, baseline_m: float) -> Optional[Vector3]:
    """Точка опорного (правого) глаза -> корпус, МЕТРЫ. None — нет глубины."""
    if not np.isfinite(p.x):
        return None
    # правый глаз -> середина стереобазы (там ось головы), мм для кинематики
    pc = ((p.x + baseline_m / 2.0) * 1000.0, p.y * 1000.0, p.z * 1000.0)
    neck = tilt = arm_kinematics.HEAD_REST
    servo = _servo
    if servo is not None:
        neck = float(servo.current_angles.get(arm_kinematics.NECK_CHANNEL, neck))
        tilt = float(servo.current_angles.get(arm_kinematics.TILT_CHANNEL, tilt))
    q = arm_kinematics.camera_to_body(pc, neck, tilt)   # мм
    return Vector3(float(q[0]) / 1000.0, float(q[1]) / 1000.0,
                   float(q[2]) / 1000.0)


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
    body = _p_to_body(p, eng.baseline)
    return body if body is not None else Vector3(float("nan"), float("nan"),
                                                 float("nan"))


# --- Захват и движение в точку кадра ---------------------------------------
GRIP_CLOSE = 180            # сомкнут (как GRIP_CLOSE в templates/webxr.html)
GRIP_OPEN = 0
GRIP_CHANNEL = {False: 8, True: 9}    # 8 — правый, 9 — левый
GRIP_SETTLE = 0.1           # с: дожимка сервы уже после того, как mover встал
GRIP_TIMEOUT = 2.0          # с: 180° хода захвата mover проходит за ~0.6 с
MOVE_TIMEOUT = 10.0         # с: дольше этого считаем, что рука не доехала
APPROACH_UP = 0.30          # м: над объектом, откуда начинается заход
GRASP_DOWN = 0.0            # м: ручная поправка "чуть ниже"; её роль теперь у TOOL_MM
ALIGN_ITERS = 3             # итераций выравнивания по маркеру
ALIGN_TOL_M = 0.004         # м: ближе этого считаем, что губки над объектом
DESCENT_STEP_M = 0.02       # м: шаг спуска в логах/диагностике
DESCENT_DT = 0.10           # с: тик непрерывного спуска (10 Гц)
DESCENT_SPEED_MPS = 0.05    # м/с: линейная скорость спуска (медленно, без рывков)
DESCENT_MEASURE_EVERY = 3   # тиков между замерами маркера (0.3 с)
DESCENT_GAIN = 0.8          # доля ошибки, которую выбираем за шаг
ROI_HALF_PX = 140           # окно поиска маркера вокруг прошлой находки
MAX_CORR_M = 0.10           # м: предел поправки по маркеру (сбой детекции)
# Точка захвата по объекту: клик часто попадает на бок/низ, брать ровно по нему
# значит толкнуть объект. Сэмплируем глубину в окне над кликом и берём центр
# верха силуэта. GRASP_DIP — насколько губки опускаются ниже верха.
# Глубина захвата от верха. Верх по сетке глубины оценивается на 10-15 мм выше
# реального (шаг 12 px, 90-й перцентиль, камера смотрит сверху), поэтому 15 мм
# оставляли губки ровно на поверхности. 30 мм даёт реальное заглубление; ниже
# стола не пускает GRASP_CLEAR.
GRASP_DIP = 0.030           # м: глубина захвата от верха объекта
GRASP_CLEAR = 0.005         # м: минимальный зазор губок над столом
GRASP_WINDOW_X = 50         # px: полуокно выборки вбок от клика
GRASP_WINDOW_UP = 100       # px: вверх от клика (объект обычно выше точки клика)
GRASP_WINDOW_DOWN = 20      # px: вниз от клика
GRASP_STEP = 12             # px: шаг сетки выборки
GRASP_NEAR = 0.20           # м: точки дальше этого от клика — фон, не объект
# Видимая поверхность объекта смещена к камере (виден бок, а не центр): тянем
# точку захвата от камеры на долю размера силуэта. 0.25 ≈ половина радиуса.
GRASP_DEEP = 0.25
GRASP_MAX_SHIFT = 0.08      # м: дальше этого верх не уводит от клика (сосед)
HAND_MASK_M = 0.09          # м: точки у своих губок (по маркеру) не считаем объектом
# Заход чуть в сторону: каждая рука идёт со СВОЕЙ стороны — правая сверху
# справа (+X корпуса), левая сверху слева (−X). Знак/величина — одна правка.
APPROACH_SIDE_M = {False: 0.012, True: -0.012}
LIFT_AFTER_UP = 0.25        # м: на сколько поднять руку после захвата/укладки
UP_MARGIN = 0.02            # м: если объект не ниже кисти на этот запас — сначала вверх
ARM_CHANNELS_POSE = (1, 2, 4, 5, 6, 7)   # каналы рук (без захватов и колёс)
# Высокоуровневые команды движения выполняются строго по одной: второй вызов
# ждёт, пока первый доведёт руку до конца (иначе IK-цели перебивают друг друга).
_motion_lock = threading.RLock()
MOVE_SPEED_DEG = 60.0       # град/с: переезды руки в задачах захвата
IK_SPEED_DEG = 40.0         # град/с: обычный IK (панель, телеоп) — было 300
_speed_ctx = [0]            # >0 — скорость задаёт _arm_slow, IK её не трогает
DESCENT_SPEED_DEG = 30.0    # град/с: заход к объекту сверху — вдвое медленнее
POSE_SPEED_DEG = 90.0       # град/с: переезд в именованную позу (mover до 300)


def _wait_motion(timeout: float = MOVE_TIMEOUT) -> bool:
    servo = _servo
    return False if servo is None else servo.wait_idle(timeout)


def _grip_settle() -> None:
    """Дождаться, пока захват реально доедет, и дать серве дожать.

    Каналы 8/9 не ограничены _arm_slow: 180° mover идёт ~0.6 с. Фиксированная
    пауза 0.3 с оставляла губки в движении на старте спуска и на подъёме —
    отсюда толчки объекта. Ждём mover и только потом трогаемся.
    """
    _wait_motion(GRIP_TIMEOUT)
    time.sleep(GRIP_SETTLE)


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
# Плечо (ch1/ch2) на время движения локтей: с ним локтю хватает места.
ELBOW_CLEAR = 100
# Поджать локти в упор к плечу перед опусканием (тесное пространство).
# Команда 10 = 170° модели; 175° в симе уже упирается губкой в плечо.
ELBOW_TUCK = 10
# ниже этого угла локтя считаем, что рука в «согнутом» состоянии (ready-подобном)
FOLDED_THRESHOLD = 115
# Ближе этого по всем трём каналам — рука уже в позе, команд ей не шлём.
POSE_TOL_DEG = 2.0


def _pose_steps(state: Optional[Dict[int, int]], dst_name: str):
    """Ступени перехода в позу dst_name от фактических углов state.

    state — последние отправленные углы каналов рук (то, где рука реально
    оказалась: после pick_point это не «поза», а произвольное положение).

    Правила:
      • перед движением локтей ставим плечо (ch1/ch2) в ELBOW_CLEAR;
      • если рука была в ready_to_grab, а едем в rest — сначала ПОДЖАТЬ локти,
        потом опустить плечи и только затем разогнуть;
      • после движения локтей плечо возвращается в требуемое значение;
      • если менять нечего — ступеней не будет.
    """
    dst = POSES[dst_name]
    state = dict(state or {})                   # что считаем текущим
    steps = []

    # Рука, которая уже в целевой позе (в пределах POSE_TOL_DEG по всем трём
    # каналам), не получает ни одной команды: датчиков нет, лишняя команда —
    # лишний рывок. Возвращаем ещё и set её каналов.
    done = set()
    for chans in arm_kinematics.ARM_CHANNELS.values():
        if all(ch in state and abs(state[ch] - dst[ch]) <= POSE_TOL_DEG
               for ch in chans.values()):
            done.update(chans.values())

    def emit(updates: Dict[int, int]) -> None:
        delta = {ch: v for ch, v in updates.items()
                 if ch not in done
                 and abs(state.get(ch, -10 ** 6) - v) > POSE_TOL_DEG}
        if delta:
            state.update(delta)
            steps.append(delta)

    # «согнута» — любая из рук, которую ещё будем двигать
    folded = any(state.get(ch, 999) < FOLDED_THRESHOLD
                 for ch in (6, 7) if ch not in done)
    if dst_name == "rest" and folded:
        # из согнутого состояния (ready или «изменённая ready») в rest:
        emit({6: ELBOW_TUCK, 7: ELBOW_TUCK})         # поджать в тесноте
        emit({4: dst[4], 5: dst[5]})                 # плечи вниз
        emit({1: ELBOW_CLEAR, 2: ELBOW_CLEAR})       # плечо под движение локтя
        emit({6: dst[6], 7: dst[7]})                 # разгиб ПОСЛЕДНИМ
        emit({1: dst[1], 2: dst[2]})                 # плечо в требуемое
    else:
        if any(state.get(ch) != dst[ch] for ch in (6, 7)):
            emit({1: ELBOW_CLEAR, 2: ELBOW_CLEAR})   # только перед движением локтей
            emit({6: dst[6], 7: dst[7]})
        emit({4: dst[4], 5: dst[5]})
        emit({6: dst[6], 7: dst[7]})
        emit({1: dst[1], 2: dst[2]})                 # плечо в требуемое
    return steps, done


def pose(name: str = POSE_DEFAULT, speed: Optional[float] = None,
         force: bool = False) -> dict:
    """Поставить робота в именованную позу и дождаться приезда.

    "rest" (по умолчанию) — руки вниз, локти выпрямлены (servo.REST_POSE,
    нулевая поза модели); "ready_to_grab" — руки сверху (плечи подняты,
    локти согнуты) — значения сняты с робота. Меняются только каналы рук:
    голова, захваты и колёса остаются как были.
    Захваты не трогаются: их ставят grab/release.
    Рука, которая уже стоит в этой позе (в пределах POSE_TOL_DEG), команд не
    получает — в результате она в списке skipped.
    speed — град/с переезда; по умолчанию POSE_SPEED_DEG.
    force — отправить позу, даже если состояние говорит, что рука уже там.
    Нужно после перезапуска: `current_angles` там = START_POSE (дефолт
    загрузки), то есть состояние врёт, а физическое положение неизвестно.
    """
    commands = POSES.get(name)
    if commands is None:
        raise ValueError(f"нет позы {name!r}; есть: {', '.join(POSES)}")
    with _motion_lock:
        return _pose_locked(name, commands,
                            POSE_SPEED_DEG if speed is None else float(speed),
                            force)


def _pose_locked(name: str, commands: Dict[int, int],
                 speed: float = POSE_SPEED_DEG,
                 force: bool = False) -> dict:
    servo = _servo
    # Отправленные углы — единственное, что мы знаем о руке (датчиков нет).
    # После pick_point это произвольное положение, поэтому «уже в этой позе»
    # определяется по углам, а не по имени последней позы.
    state = (None if force else
             {ch: int(servo.current_angles.get(ch, 90)) for ch in ARM_CHANNELS_POSE}
             if servo is not None else None)
    steps, done = _pose_steps(state, name)    # переходы работают всегда
    sent = {}
    arrived = True
    with _arm_slow(speed):                   # поза ставится без рывка
        for step in steps:
            for channel, angle in step.items():
                if set_servo_command(channel, int(angle)):
                    sent[channel] = int(angle)
            arrived = _wait_motion() and arrived   # доводим ступень до конца
    skipped = [side for side, chans in arm_kinematics.ARM_CHANNELS.items()
               if all(ch in done for ch in chans.values())]
    return {"ok": servo is not None, "pose": name, "servo": sent,
            "arrived": arrived, "skipped": skipped}


def _system_command(command: str) -> dict:
    """Сначала rest, потом poweroff/reboot всей системы.

    Порядок обязателен: питание снимается только с руками в rest (иначе сервы
    отпускают руку там, где она стояла). Ждём приезда, даём сервам 0.3 с дожать
    и только потом командуем systemd. Процесс не ждёт выключения — команда
    уходит своей сессией, чтобы systemd не убил её вместе с сервисом.
    """
    with _motion_lock:
        posed = pose("rest")
    log(f"[{command}] pose(rest): {posed}")
    print(f"[{command}] pose(rest): {posed}", flush=True)
    time.sleep(0.3)                       # серве физически дожать
    try:
        subprocess.Popen(["systemctl", command], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = True
    except OSError as exc:                # нет systemctl / нет прав
        log(f"[{command}] ошибка: {exc}")
        ok = False
    return {"ok": ok, "command": command, "pose": posed}


def poweroff() -> dict:
    """Поставить руки в rest и выключить робота целиком (systemctl poweroff)."""
    return _system_command("poweroff")


def restart() -> dict:
    """Поставить руки в rest и перезагрузить робота целиком (systemctl reboot)."""
    return _system_command("reboot")


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
    _speed_ctx[0] += 1
    try:
        yield
    finally:
        _speed_ctx[0] -= 1
        # Лимит снимаем ПОСЛЕ приезда: иначе mover на остатке пути разгоняется
        # до 300 °/с — спуск шёл «бросками» (рывок, пауза, рывок).
        servo.wait_idle(MOVE_TIMEOUT)
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


_marker_roi: Dict[bool, Optional[Tuple[int, int, int, int]]] = {False: None,
                                                               True: None}


def calibrate_depth_scale(left: bool = False) -> dict:
    """Подстроить масштаб глубины по маркеру — прямо по ходу работы.

    Маркер 38 мм даёт абсолютную дистанцию (PnP), не зависящую от привязки
    глубины. Сравниваем её с глубиной в пикселе маркера и правим масштаб
    движка. Вызывается перед каждым захватом, пока маркер виден.
    """
    frame = _stereo_frame()
    if frame is None:
        return {"ok": False, "reason": "нет стерео-кадра"}
    left_img, right_img = split_stereo(frame)
    if right_img is None:
        return {"ok": False, "reason": "нет стерео-кадра"}
    eng = _stereo_depth()
    marker_id = gripper_marker.MARKER_ID[bool(left)]
    found = gripper_marker.detect(right_img, eng.K[1], eng.D[1],
                                  eng._Rrect[1], eng._P[1], ids=(marker_id,))
    marker = found.get(marker_id)
    if marker is None:
        return {"ok": False, "reason": "маркер не виден"}
    cu, cv = np.clip(np.round(marker["center"]).astype(int), 0,
                     np.array(right_img.shape[:2])[::-1] - 1)
    maps = eng._maps[1]
    raw = (float(maps[0][cv, cu]), float(maps[1][cv, cu]))
    p = eng.point_at(left_img, right_img, *raw, eye="right")
    if not np.isfinite(p.x):
        return {"ok": False, "reason": "нет глубины в маркере"}
    pnp = float(np.linalg.norm(np.asarray(marker["tvec"], float)))
    got = float(np.linalg.norm([p.x, p.y, p.z]))
    if pnp > 1e-3 and got > 1e-3:
        eng.scale *= pnp / got
    return {"ok": True, "ratio": round(got / max(pnp, 1e-6), 3),
            "scale": round(eng.scale, 3), "depth_m": round(got, 3),
            "marker_m": round(pnp, 3)}


def marker_calibrate(x: float, y: float, left: bool = False) -> dict:
    """Смещение маркер→губки по РЕАЛЬНОЙ точке кадра, а не по FK.

    Поставь руку телеопом так, чтобы губки стояли ровно над точкой, и укажи
    её в кадре (доли 0..1). Возвращает v (мм) в системе маркера — его надо
    прописать в gripper_marker.GRASP_FROM_MARKER_MM. Так калибровка не
    наследует ошибку кинематики.
    """
    frame = _stereo_frame()
    if frame is None:
        return {"ok": False, "reason": "нет стерео-кадра"}
    left_img, right_img = split_stereo(frame)
    if right_img is None:
        return {"ok": False, "reason": "нет стерео-кадра"}
    eng = _stereo_depth()
    marker_id = gripper_marker.MARKER_ID[bool(left)]
    found = gripper_marker.detect(right_img, eng.K[1], eng.D[1],
                                  eng._Rrect[1], eng._P[1], ids=(marker_id,))
    marker = found.get(marker_id)
    if marker is None:
        return {"ok": False, "reason": "маркер не виден"}
    body = coords(x, y)
    if not np.isfinite(body.x):
        return {"ok": False, "reason": "нет глубины в точке"}
    F = np.diag([1.0, -1.0, 1.0])          # камера OpenCV -> корпус по Y
    cam = np.array([eng.baseline / 2.0, 0.100, 0.060])
    P = F @ (np.array([body.x, body.y, body.z]) - cam)
    R = cv2.Rodrigues(np.asarray(marker["rvec"], float))[0]
    t = np.asarray(marker["tvec"], float).reshape(3)
    v = 1000.0 * (R.T @ (P - t))
    if float(np.linalg.norm(v)) < 40.0:
        return {"ok": False,
                "reason": ("точка попала в губки/маркер: |v| < 40 мм, а губки "
                           "не могут быть у центра маркера — кликни ОБЪЕКТ "
                           "между губками"),
                "v_mm": [round(float(x), 1) for x in v], "point": body}
    return {"ok": True, "v_mm": [round(float(x), 1) for x in v],
            "now_mm": [round(float(x), 1) for x in gripper_marker.GRASP_FROM_MARKER_MM],
            "point": body}


def _marker_grasp(left: bool = False) -> Optional[Vector3]:
    """Точка захвата по ArUco-маркеру этой руки, м в системе корпуса.

    Маркер ищется в ПРАВОМ глазу (опорный), поэтому в корпус — та же цепочка,
    что у coords(): +полубаза, поворот головы, монтаж. Ни одного лишнего
    перевода между глазами.
    """
    frame = _stereo_frame()
    if frame is None:
        return None
    _left_img, right_img = split_stereo(frame)
    if right_img is None:
        return None
    eng = _stereo_depth()
    marker_id = gripper_marker.MARKER_ID[bool(left)]
    roi = _marker_roi[bool(left)]
    found = gripper_marker.detect(right_img, eng.K[1], eng.D[1],
                                  eng._Rrect[1], eng._P[1], ids=(marker_id,),
                                  roi=roi)
    if not found and roi is not None:            # потеряли — ищем по всему кадру
        _marker_roi[bool(left)] = None
        found = gripper_marker.detect(right_img, eng.K[1], eng.D[1],
                                      eng._Rrect[1], eng._P[1], ids=(marker_id,))
    marker = found.get(marker_id)
    if marker is None:
        return None
    cx, cy = marker["center"]
    _marker_roi[bool(left)] = (int(cx) - ROI_HALF_PX, int(cy) - ROI_HALF_PX,
                               int(cx) + ROI_HALF_PX, int(cy) + ROI_HALF_PX)
    grasp, _center = gripper_marker.grasp_from_marker(marker)
    return _p_to_body(Vector3(float(grasp[0]), float(grasp[1]), float(grasp[2])),
                      eng.baseline)


def _align_by_marker(target: Vector3, left: bool = False,
                     iters: int = ALIGN_ITERS,
                     tol: float = ALIGN_TOL_M) -> dict:
    """Подвести губки в XY к target по маркеру. Высоту не трогаем.

    Ошибка = (фактическая точка захвата из маркера) − target. Компенсируем её
    сдвигом команды на ту же величину (ошибка модели почти постоянна), 2-3
    итерации. Возвращает уточнённую точку команды.
    """
    base = Vector3(target.x, target.y, target.z)
    cmd = Vector3(target.x, target.y, target.z)
    error = None
    for _ in range(max(1, iters)):
        _move_to(_ee_target(cmd, left), left)
        _wait_motion()
        grasp = _marker_grasp(left)
        if grasp is None:
            return {"ok": False, "reason": "маркер не виден",
                    "point": cmd, "error_mm": None}
        ex, ez = grasp.x - base.x, grasp.z - base.z
        error = float(np.hypot(ex, ez))
        if error <= tol:
            return {"ok": True, "point": cmd, "error_mm": error * 1000.0}
        cmd = Vector3(cmd.x - ex, cmd.y, cmd.z - ez)
    return {"ok": bool(error is not None and error <= tol), "point": cmd,
            "error_mm": None if error is None else error * 1000.0}


def align_xy(x: float, y: float, left: bool = False) -> dict:
    """Выровнять губки по видимой точке кадра (X/Z) с помощью маркера.

    Рука остаётся на текущей высоте; правится только XY. Если маркера не видно
    — возвращает ok=False, ничего не двигая.
    """
    with _motion_lock:
        point = coords(x, y)
        if not np.isfinite(point.x):
            return {"ok": False, "reason": "нет глубины в этой точке кадра"}
        current = _current_point(left)
        height = current.y if current is not None else point.y + APPROACH_UP
        with _arm_slow(MOVE_SPEED_DEG):
            return _align_by_marker(Vector3(point.x, height, point.z), left)


def depth_check(left: bool = False) -> dict:
    """Сверить глубину по привязке с дистанцией до маркера (PnP, без привязки).

    Маркер — чёрный квадрат 38 мм: его tvec даёт расстояние независимо от
    формулы глубины.
    Сравниваем с |point_at| в пикселе маркера. Расхождение больше 5% значит
    привязку глубины надо пересобрать (`charuco_calib.py --depth-fit`).
    """
    frame = _stereo_frame()
    if frame is None:
        return {"ok": False, "reason": "нет стерео-кадра"}
    left_img, right_img = split_stereo(frame)
    if right_img is None:
        return {"ok": False, "reason": "нет стерео-кадра"}
    eng = _stereo_depth()
    marker_id = gripper_marker.MARKER_ID[bool(left)]
    found = gripper_marker.detect(right_img, eng.K[1], eng.D[1], eng._Rrect[1],
                                  eng._P[1], ids=(marker_id,))
    marker = found.get(marker_id)
    if marker is None:
        return {"ok": False, "reason": "маркер не виден"}
    cu, cv = np.clip(np.round(marker["center"]).astype(int), 0,
                     np.array(right_img.shape[:2])[::-1] - 1)
    maps = eng._maps[1]                    # выпрямленный пиксель -> сырой
    raw = (float(maps[0][cv, cu]), float(maps[1][cv, cu]))
    point = eng.point_at(left_img, right_img, *raw, eye="right")
    if not np.isfinite(point.x):
        return {"ok": False, "reason": "нет глубины в маркере"}
    pnp = float(np.linalg.norm(np.asarray(marker["tvec"], float)))
    got = float(np.linalg.norm([point.x, point.y, point.z]))
    return {"ok": True, "left": bool(left), "depth_m": round(got, 3),
            "marker_m": round(pnp, 3), "ratio": round(got / max(pnp, 1e-6), 3),
            "verdict": "ok" if abs(got / max(pnp, 1e-6) - 1.0) < 0.05 else
                       "привязку пересобрать (--depth-fit)"}


def marker_check(left: bool = False) -> dict:
    """Сравнить точку захвата из маркера с моделью (FK + TOOL_MM).

    Возвращает ошибку модели по осям и готовое значение TOOL_MM: если рука
    уходит от объекта на E, то TOOL_MM += E.
    """
    grasp = _marker_grasp(left)
    if grasp is None:
        return {"ok": False, "reason": "маркер не виден"}
    current = _current_point(left)
    if current is None:
        return {"ok": False, "reason": "нет сервоприводов"}
    offset = arm_kinematics.tool_offset(left) / 1000.0
    model = Vector3(current.x + offset[0], current.y + offset[1],
                    current.z + offset[2])
    delta = [grasp.x - model.x, grasp.y - model.y, grasp.z - model.z]
    suggest = [arm_kinematics.TOOL_MM['left' if left else 'right'][i] + delta[i] * 1000.0
               for i in range(3)]
    return {"ok": True, "left": bool(left), "marker": grasp, "model": model,
            "delta_mm": [round(v * 1000.0, 1) for v in delta],
            "suggest_tool_mm": [round(float(v), 1) for v in suggest]}


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


def _ee_target(grasp: Vector3, left: bool = False) -> Vector3:
    """Точка захвата -> цель для IK (EE), с поправкой на смещение губок."""
    ee = arm_kinematics.ee_for_grasp(
        (grasp.x * 1000.0, grasp.y * 1000.0, grasp.z * 1000.0), left=left)
    return Vector3(ee[0] / 1000.0, ee[1] / 1000.0, ee[2] / 1000.0)


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


def _grasp_point(x: float, y: float, left: bool = False) -> dict:
    """Точка захвата по ОБЪЕКТУ под кликом: центр и верх локального силуэта.

    Клик может быть на боку, на нижней части или на краю — если тянуться ровно
    туда, губки бьют объект сбоку. Поэтому: берём клик как подсказку, сэмплируем
    глубину сеткой (вверх больше, чем вниз), отбрасываем фон (дальше GRASP_NEAR
    от клика) и берём ВЕРХ оставшихся точек. Центр верха по X/Z — ось губок,
    высота — верх минус GRASP_DIP, но не ниже стола (низ точек) + GRASP_CLEAR.

    Возвращает {ok, point, top, bottom, size, samples, reason}. Ничего не двигает.
    """
    eng, left_img, right_img, px, py = _depth_sample(x, y)
    if left_img is None:
        return {"ok": False, "reason": "нет стерео-кадра"}
    eye = _shown_eye()
    h, w = left_img.shape[:2]
    anchor = _p_to_body(eng.point_at(left_img, right_img, px, py, eye=eye),
                        eng.baseline)
    if anchor is None:                 # клик без глубины — тянемся по модели
        return {"ok": False, "reason": "нет глубины в точке клика"}
    # своя кисть в кадре (поднесена к объекту) — не путать её с объектом
    hand = _marker_grasp(left)   # left — сторона руки, не кадр
    hand_r2 = HAND_MASK_M ** 2
    pts = []
    for dy in range(-GRASP_WINDOW_UP, GRASP_WINDOW_DOWN + 1, GRASP_STEP):
        for dx in range(-GRASP_WINDOW_X, GRASP_WINDOW_X + 1, GRASP_STEP):
            u, v = px + dx, py + dy
            if not (2 <= u < w - 2 and 2 <= v < h - 2):
                continue
            p = _p_to_body(eng.point_at(left_img, right_img, u, v, eye=eye),
                           eng.baseline)
            if p is None:
                continue
            if (abs(p.x - anchor.x) <= GRASP_NEAR
                    and abs(p.y - anchor.y) <= GRASP_NEAR
                    and abs(p.z - anchor.z) <= GRASP_NEAR
                    and (hand is None or (p.x - hand.x) ** 2 + (p.y - hand.y) ** 2
                         + (p.z - hand.z) ** 2 > hand_r2)):
                pts.append(p)
    if len(pts) < 6:
        return {"ok": False, "reason": "мало точек глубины вокруг", "anchor": anchor}
    top = float(np.percentile([p.y for p in pts], 90))    # выбросы не в счёт
    upper = [p for p in pts if p.y >= top - 0.010]
    if len(upper) < 3:
        upper = sorted(pts, key=lambda p: p.y)[-3:]
    gx = float(np.median([p.x for p in upper]))
    gz = float(np.median([p.z for p in upper]))
    if float(np.hypot(gx - anchor.x, gz - anchor.z)) > 0.08:
        gx, gz = anchor.x, anchor.z    # верх чужого объекта — остаёмся над кликом
    bottom = float(np.percentile([p.y for p in pts], 10))
    span_x = float(np.percentile([p.x for p in upper], 90)
                   - np.percentile([p.x for p in upper], 10))
    span_z = float(np.percentile([p.z for p in upper], 90)
                   - np.percentile([p.z for p in upper], 10))
    size = max(span_x, span_z)
    # От камеры к центру силуэта: видимая поверхность всегда ближе к камере,
    # поэтому без этого губки наводятся на край и промахиваются.
    ux, uz = gx, gz - arm_kinematics.HEAD_LENS_Z / 1000.0
    norm = float(np.hypot(ux, uz))
    if norm > 1e-3:
        gx += ux / norm * GRASP_DEEP * size
        gz += uz / norm * GRASP_DEEP * size
    dip = min(GRASP_DIP, max(0.0, top - bottom - GRASP_CLEAR))
    return {"ok": True, "point": Vector3(gx, top - dip, gz), "top": top,
            "bottom": bottom, "size": size,
            "samples": len(pts), "anchor": anchor}


def _correct_cmd(want: Vector3, cmd: Vector3, current: Vector3,
                 left: bool = False):
    """Поправка команды по маркеру: (новый cmd, ошибка модели, остаток до цели).

    Ошибка модели = реальные губки − КОМАНДА (не цель!): на диагональном заходе
    рука ещё не над объектом, и сравнение с целью принимало бы путь за ошибку.
    Модель почти постоянна, поэтому сдвигаем цель на неё же; остаток до цели
    (факт − want) — для диагностики.
    """
    grasp = _marker_grasp(left)
    if grasp is None:
        return cmd, None, None
    ex, ez = grasp.x - current.x, grasp.z - current.z
    model = float(np.hypot(ex, ez))
    residual = float(np.hypot(grasp.x - want.x, grasp.z - want.z))
    # Сдвиг цели — сходимся к «команда = цель − ошибка модели»:
    # dx += g·(−ex − dx), иначе при постоянной ошибке поправка копится без предела.
    dx = (cmd.x - want.x) + DESCENT_GAIN * (-ex - (cmd.x - want.x))
    dz = (cmd.z - want.z) + DESCENT_GAIN * (-ez - (cmd.z - want.z))
    dx = max(-MAX_CORR_M, min(MAX_CORR_M, dx))
    dz = max(-MAX_CORR_M, min(MAX_CORR_M, dz))
    return Vector3(want.x + dx, want.y, want.z + dz), model, residual


def _dist(a: Vector3, b: Vector3) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _reachable_at(point: Vector3, left: bool = False,
                  tol: float = 0.008) -> bool:
    """Достижима ли ТОЧКА (м, корпус) с ошибкой не больше tol."""
    ee = _ee_target(point, left)
    base, _ = _reachable(ee, left)
    return base is not None and _dist(base, ee) <= tol


def _approach_point(point: Vector3, left: bool = False,
                    distance: float = APPROACH_UP) -> Vector3:
    """Точка старта захода: НЕ БЛИЖЕ `distance` от объекта, лучше сверху.

    Строго вертикально рука поднимается невысоко (предплечье 220, плечо 330),
    поэтому перебираем наклон от вертикали в свою сторону (правая — вправо,
    левая — влево) и к корпусу, пока точка на нужной дистанции не станет
    достижимой: телеоп такие точки берёт, значит они реальны.
    """
    side = 1.0 if not left else -1.0
    for dist in (distance, distance * 0.8, max(0.10, distance * 0.5)):
        for tilt in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0):
            t = np.radians(tilt)
            for az in (90.0, 60.0, 120.0, 0.0, 180.0):   # своя сторона, потом к корпусу
                a = np.radians(az)
                d = np.array([np.sin(t) * np.sin(a) * side,
                              np.cos(t),
                              -np.sin(t) * np.cos(a)]) * dist
                cand = Vector3(point.x + float(d[0]), point.y + float(d[1]),
                               point.z + float(d[2]))
                if _reachable_at(cand, left):
                    return cand
    y = point.y + distance                  # крайний случай: сколько достанем вверх
    while y > point.y + 0.02:
        cand = Vector3(point.x, y, point.z)
        if _reachable_at(cand, left):
            return cand
        y -= 0.02
    return Vector3(point.x, point.y + 0.02, point.z)


def _approach(point: Vector3, left: bool = False,
              align: bool = True, lateral: bool = True,
              side_m: Optional[float] = None) -> dict:
    """Заход сверху 30 см, затем спуск шагами с коррекцией по маркеру.

    point — точка ЗАХВАТА (м, корпус). Калибровка кисти не нужна: на каждом
    шаге спуска после остановки сравниваем реальную точку захвата (из маркера)
    с точкой объекта и сдвигаем команду на ошибку; ошибка модели почти
    постоянна, поэтому сходится за 2-3 шага. Внизу — ещё одна поправка (только
    XY). Если маркера не видно — идём по модели.
    lateral — сместить весь заход вбок на APPROACH_SIDE_M, чтобы неподвижный
    палец не задел объект; смещение остаётся и в точке захвата.
    """
    side = (APPROACH_SIDE_M[bool(left)] if side_m is None else side_m) \
        if lateral else 0.0
    want = Vector3(point.x + side, point.y, point.z)   # куда должны попасть губки
    cmd = Vector3(want.x, want.y, want.z)              # куда ведём руку (с поправкой)
    # Старт захода — не ближе APPROACH_UP от объекта, сверху или сверху-сбоку.
    current = _approach_point(want, left)
    start_point = current
    corrections = 0
    seen = 0
    residual = None
    # ОДИН непрерывный проход: лимит скорости ставим на весь спуск, цель
    # подвигаем каждые 0.1 с. Раньше каждый шаг был отдельным _arm_slow с
    # ожиданием остановки — отсюда «рывки».
    with _arm_slow(DESCENT_SPEED_DEG):
        with _motion_lock:
            _move_to(_ee_target(current, left), left)
        for tick in range(1, 400):
            dx, dy, dz = cmd.x - current.x, cmd.y - current.y, cmd.z - current.z
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if dist <= 1e-6:
                break
            k = min(1.0, DESCENT_SPEED_MPS * DESCENT_DT / dist)
            current = Vector3(current.x + dx * k, current.y + dy * k,
                              current.z + dz * k)
            with _motion_lock:
                result = _move_to(_ee_target(current, left), left)
            if align and tick % DESCENT_MEASURE_EVERY == 0:
                time.sleep(DESCENT_DT)
                new_cmd, model, residual = _correct_cmd(want, cmd, current, left)
                if model is not None:
                    seen += 1
                    corrections += int(_dist(new_cmd, cmd) > 0.001)
                cmd = new_cmd
            else:
                time.sleep(DESCENT_DT)
        _wait_motion()
    if align:                              # внизу — последняя поправка, только XY
        cmd, model, residual = _correct_cmd(want, cmd, current, left)
        if model is not None:
            seen += 1
            if model > ALIGN_TOL_M:
                with _arm_slow(DESCENT_SPEED_DEG):
                    _move_to(_ee_target(cmd, left), left)
                    _wait_motion()
                corrections += 1
    result["target"] = cmd
    result["approach"] = start_point
    result["align"] = {"ok": seen > 0, "seen": seen,
                       "corrections": corrections,
                       "residual_mm": None if residual is None
                       else round(residual * 1000.0, 1),
                       "reason": "нет маркера" if seen == 0 else "ok"}
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
        result = _move_to(_ee_target(point, left), left)
        result["point"] = point
        result["arrived"] = bool(result.get("moved")) and _wait_motion()
    return result


def pick_point(x: float, y: float, left: bool = False) -> dict:
    """Взять объект под точкой кадра (x, y от 0 до 1).

    Раскрыть захват -> точка берётся по СИЛУЭТУ объекта (верх и центр верха,
    а не по клику — иначе губки бьют предмет сбоку) -> заход на 30 см выше ->
    спуск шагами с коррекцией по маркеру -> сжать -> поднять на LIFT_AFTER_UP.
    Заход смещён вбок (APPROACH_SIDE_M), чтобы неподвижный палец не задел объект.
    Результат: point — куда реально целились, grasp — как найдена точка.
    """
    return _point_action(x, y, GRIP_OPEN, GRIP_CLOSE, left=left, lift_after=True)


def place_point(x: float, y: float, left: bool = False) -> dict:
    """Положить: сжать захват -> зайти выше -> медленно спуститься -> разжать
    -> подняться на 20 см.
    """
    return _point_action(x, y, GRIP_CLOSE, GRIP_OPEN, left=left, lift_after=True)


def _point_action(x: float, y: float, grip_before: int, grip_after: int,
                  left: bool = False, lift_after: bool = False) -> dict:
    with _motion_lock:
        return _point_action_locked(x, y, grip_before, grip_after,
                                    lift_after=lift_after, left=left)


def _point_action_locked(x: float, y: float, grip_before: int, grip_after: int,
                         lift_after: bool = False, left: bool = False) -> dict:
    channel = GRIP_CHANNEL[bool(left)]
    set_servo_command(channel, grip_before)   # сначала клешня, потом движение
    _grip_settle()                            # и кадр для цели — уже без неё
    # Захват: точка по силуэту объекта (верх + центр), а не по клику — так
    # губки не бьют объект сбоку. Укладка: ровно точка клика.
    # Захват: сначала подстроить масштаб глубины по маркеру (эталон 38 мм),
    # потом искать точку по силуэту — иначе и глубина, и высота врут.
    scale_info = calibrate_depth_scale(left) if grip_before == GRIP_OPEN else None
    grasp_info = _grasp_point(x, y, left) if grip_before == GRIP_OPEN else None
    point = (grasp_info["point"] if grasp_info and grasp_info.get("ok")
             else coords(x, y))
    if not np.isfinite(point.x):
        return {"ok": False, "moved": False, "arrived": False,
                "message": "нет глубины в этой точке кадра",
                "point": point, "grasp": grasp_info}
    if GRASP_DOWN:                     # ручная поправка (обычно 0 — см. TOOL_MM)
        point = Vector3(point.x, point.y - GRASP_DOWN, point.z)
    # IK двигает конец цепи: пересчитываем в EE с учётом смещения губок
    base, ik = _reachable(_ee_target(point, left), left)
    if base is None:
        return {"ok": False, "moved": False, "arrived": False,
                "message": ik.get("message") or "IK не нашёл решения",
                "point": point}
    if base.z < 0.05:                        # ближе плоскости корпуса
        return {"ok": False, "moved": False, "arrived": False,
                "message": "объект вплотную к корпусу (z < 5 см) — не тянемся",
                "point": point, "reachable": base}
    with _arm_slow(MOVE_SPEED_DEG):
        _lift_if_needed(point, left)     # объект выше/на уровне кисти — сначала вверх
    # Боковой заход: для мелкого объекта смещение меньше — иначе он окажется
    # на самом краю губок и его снесёт движущимся пальцем.
    side = None
    if grasp_info and grasp_info.get("ok"):
        base_side = APPROACH_SIDE_M[bool(left)]
        side = float(np.sign(base_side) * min(abs(base_side),
                                              0.25 * float(grasp_info["size"])))
    result = _approach(point, left, lateral=grip_before == GRIP_OPEN, side_m=side)
    # Захват переключаем после всех движений и независимо от того, дошла ли рука
    # до цели: оставить клешню в прежнем состоянии хуже, чем сжать/разжать.
    set_servo_command(channel, grip_after)
    _grip_settle()                            # сжали — только теперь поднимаем
    grasp = result.get("target", point)
    if lift_after:
        with _arm_slow(MOVE_SPEED_DEG):
            _move_to(_ee_target(Vector3(grasp.x, grasp.y + LIFT_AFTER_UP,
                                        grasp.z), left), left)
            result["arrived"] = _wait_motion()
    result["point"] = point
    result["grasp"] = grasp_info
    result["depth_scale"] = scale_info
    result["reachable"] = base
    result["grip"] = grip_after
    return result


def set_ik_branch(name: str) -> str:
    """Предпочитаемая ветка IK: "top" (сверху) или "bottom" (снизу)."""
    return arm_kinematics.set_branch(name)


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
    if _speed_ctx[0] == 0 and _servo is not None:
        # Прямой IK (панель/телеоп) — не на 300 °/с: ставим спокойную скорость.
        _servo.set_speed_limit(ARM_CHANNELS_POSE, IK_SPEED_DEG)
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

