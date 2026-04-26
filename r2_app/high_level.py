import subprocess
import threading
import time

from r2_app.bootstrap import build_services
from r2_app.web import create_app

_services = None
_app = None
_services_lock = threading.Lock()


def _ensure_services():
    global _services, _app
    if _services is not None:
        return _services
    with _services_lock:
        if _services is None:
            _services = build_services()
            _app = create_app(services=_services, logger=_services.logger)
    return _services


def _hardware_worker():
    services = _ensure_services()
    services.logger.log("Инициализация hardware в отдельном потоке")
    services.hardware.initialize()
    services.shell.start()
    services.logger.log("Hardware/shell готовы")


def _start_kiosk_browser(url="http://localhost"):
    cmd = [
        "chromium",
        "--kiosk",
        "--noerrdialogs",
        "--disable-infobars",
        f"--app={url}"
    ]

    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Браузер запущен по адресу: {url}")
    except FileNotFoundError:
        print("Ошибка: Chromium не найден. Убедитесь, что он установлен (sudo apt install chromium).")

def _web_worker():
    services = _ensure_services()
    services.logger.log("Web сервер запускается в отдельном потоке")
    _start_kiosk_browser(f"http://{services.config.host}:{services.config.http_port}/screen")
    _app.run(host=services.config.host, port=services.config.http_port, debug=False, threaded=True, use_reloader=False)


def _start_background_threads():
    threading.Thread(target=_hardware_worker, daemon=True, name="r2-hardware-thread").start()
    threading.Thread(target=_web_worker, daemon=True, name="r2-web-thread").start()


def get_stereo_camera():
    return _ensure_services().hardware.camera


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


def angle(servo: int, angle_value: int):
    servo_controller = _ensure_services().hardware.servo
    if servo_controller is None:
        return False
    return servo_controller.set_servo(servo, angle_value, smooth=True, step_delay=0.01, step_angle=2)


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
    _ensure_services().hardware.servo_tracking_enabled = bool(enabled)


def gemini_test(prompt: str = "Ping from R2"):
    return _ensure_services().gemini.test(prompt=prompt)


def emote(emotion_name: str):
    return _ensure_services().emote.set_emote(emotion_name)


def set_eyes_position(x, y):
    _ensure_services().emote.set_eyes_position(x, y)


def start():
    # Backward-compatible blocking runner.
    start_background()
    while True:
        time.sleep(0.01)


def start_background():
    _ensure_services()
    _start_background_threads()

