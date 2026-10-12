import json
import os
import platform
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple, Union

import numpy as np

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

MAX_FRAME_SIZE = 4 * 1024 * 1024

# Watchdog камеры: если кадров нет дольше STALL_SEC, пробуем восстановить
# захват (release+reinit), а затем — программный USB-сброс (см. usb_reset).
WATCHDOG_PERIOD = 2.0
STALL_SEC = 3.0


class CameraInitError(Exception):
    pass


class StereoCamera:
    def __init__(
        self,
        camera_param_file: str = "cam_params.json",
        source: Union[int, str] = 0,
        capture_width: int = 2560,
        capture_height: int = 720,
        backend: str = "ffmpeg",
        fps_target: int = 30,
        simulation: bool = False,
    ):
        self.camera_param_file: str = camera_param_file
        self.camera_source: Union[int, str] = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock: threading.Lock = threading.Lock()
        # Симуляция: если настоящей камеры нет — синтетический кадр SIMULATION.
        self.simulation: bool = simulation
        self._sim_active: bool = False
        self._sim_tick: int = 0

        self.img_size: Tuple[int, int] = (640, 360)
        self.eye_w: int = 1280
        self.eye_h: int = 720
        self.show_left: bool = False      # по умолчанию — ПРАВЫЙ глаз
        self.fps: float = 30.0
        self._last_frame_time: float = time.time()
        self._frame_count: int = 0

        self.capture_width: int = capture_width
        self.capture_height: int = capture_height
        self.backend: str = backend
        self.fps_target: int = fps_target
        self.cam_fov_h: float = 120.0  # горизонтальный FOV камеры, ° (фишай)
        self.actual_width: int = 0
        self.actual_height: int = 0
        self._eye_x0: int = 0
        self._eye_x1: int = self.eye_w
        self._eye_crop_w: int = self.eye_w
        self._eye_layout_override: bool = False
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_seq: int = 0
        self._raw_jpeg: Optional[bytes] = None
        self._latest_stereo_jpeg: Optional[bytes] = None
        self._stereo_seq: int = 0
        # Число активных MJPEG-потребителей: пока никто не смотрит, не тратим
        # CPU на decode+encode кадра (на Orange Pi это заметная доля ядра).
        self._feed_consumers: int = 0
        self._stereo_consumers: int = 0
        self._capture_thread: Optional[threading.Thread] = None
        self._eye_thread: Optional[threading.Thread] = None
        self._capture_running: bool = False
        # Watchdog: время последнего кадра (monotonic), флаг восстановления
        # и «желаемый» бэкенд, который восстанавливаем после USB-сброса.
        self._last_frame_mono: float = time.monotonic()
        self._recovering: bool = False
        self._watchdog_thread: Optional[threading.Thread] = None
        self._preferred_backend: str = backend

        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._pipe = None
        self._jpeg_remainder = bytearray()
        self._ffmpeg_failures: int = 0

        try:
            with open(self.camera_param_file, encoding="utf-8") as f:
                params = json.load(f)
            self.capture_width = int(params.get("capture_width", self.capture_width))
            self.capture_height = int(params.get("capture_height", self.capture_height))
            self.backend = str(params.get("camera_backend", self.backend))
            self._preferred_backend = self.backend
            self.fps_target = int(params.get("fps_target", self.fps_target))
            src = params.get("camera_source")
            if isinstance(src, str) and src:
                self.camera_source = src
            elif isinstance(src, int):
                self.camera_source = src
            isize = params.get("img_size")
            if isinstance(isize, (list, tuple)) and len(isize) >= 2:
                self.img_size = (int(isize[0]), int(isize[1]))
            im_size = params.get("imSize")
            if isinstance(im_size, (list, tuple)) and len(im_size) >= 2:
                self.eye_w = int(im_size[0])
                self.eye_h = int(im_size[1])
            fov_h = params.get("cam_fov_h")
            try:
                if fov_h is not None:
                    self.cam_fov_h = float(fov_h)
            except (TypeError, ValueError):
                pass
            el = params.get("eye_layout")
            if isinstance(el, dict):
                try:
                    ox0 = int(el.get("x0", -1))
                    ox1 = int(el.get("x1", -1))
                    ow = int(el.get("w", -1))
                    if ox0 >= 0 and ox1 > ox0 and ow > 0:
                        self._eye_x0 = ox0
                        self._eye_x1 = ox1
                        self._eye_crop_w = ow
                        self._eye_layout_override = True
                        print(f"[Camera] eye_layout override: left x={ox0}, "
                              f"right x={ox1}, eye w={ow}", flush=True)
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass

    def initialize_camera(self) -> bool:
        if self._sim_active:
            return True
        if self.backend == "ffmpeg" and platform.system() != "Windows":
            try:
                if self._start_ffmpeg():
                    self.actual_width = self.capture_width
                    self.actual_height = self.capture_height
                    return True
                print("[Camera] FFmpeg backend failed, falling back to OpenCV", flush=True)
            except Exception as e:
                print(f"[Camera] FFmpeg backend error: {e}", flush=True)
        self.backend = "opencv"
        try:
            ok = self._init_opencv()
        except CameraInitError as exc:
            print(f"[Camera] OpenCV init error: {exc}", flush=True)
            ok = False
        if not ok and self.simulation:
            self._start_simulation()
            return True
        return ok

    def _start_simulation(self) -> None:
        """Настоящей камеры нет — синтетические кадры с надписью SIMULATION.

        backend выставляем в ffmpeg-путь: sim-луп публикует _raw_jpeg,
        а читатели (get_rectified_pair и др.) идут через _decode_latest.
        """
        self._sim_active = True
        self.backend = "ffmpeg"
        self.actual_width = self.capture_width
        self.actual_height = self.capture_height
        print(f"[Camera] SIMULATION: камеры нет — синтетический кадр "
              f"{self.capture_width}x{self.capture_height}", flush=True)

    def _init_opencv(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                return False
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if platform.system() == "Linux":
                try:
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
                except Exception:
                    pass
            return True
        except Exception as e:
            raise CameraInitError(f"Failed to open camera: {e}") from e

    def _start_ffmpeg(self) -> bool:
        if shutil.which("ffmpeg") is None:
            return False
        if isinstance(self.camera_source, str):
            dev = self.camera_source
        else:
            dev = f"/dev/video{int(self.camera_source)}"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "v4l2", "-input_format", "mjpeg",
            "-video_size", f"{self.capture_width}x{self.capture_height}",
            "-framerate", str(self.fps_target),
            "-i", dev,
            "-c:v", "copy",
            "-f", "mjpeg", "pipe:1",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        time.sleep(0.3)
        if proc.poll() is not None:
            try:
                _, err = proc.communicate(timeout=1)
                err_text = err.decode(errors="replace").strip()
                if err_text:
                    print(f"[Camera] ffmpeg exited: {err_text}", flush=True)
            except Exception:
                pass
            return False
        self._ffmpeg_proc = proc
        self._pipe = proc.stdout
        self._jpeg_remainder = bytearray()
        threading.Thread(
            target=self._ffmpeg_stderr_reader, args=(proc,),
            daemon=True, name="ffmpeg-stderr",
        ).start()
        return True

    def _ffmpeg_stderr_reader(self, proc: subprocess.Popen) -> None:
        try:
            for line in iter(proc.stderr.readline, b""):
                text = line.decode(errors="replace").strip()
                if text:
                    print(f"[ffmpeg] {text}", flush=True)
        except Exception:
            pass

    def _stop_ffmpeg(self) -> None:
        proc, self._ffmpeg_proc = self._ffmpeg_proc, None
        self._pipe = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _extract_frame(self, buf: bytearray):
        soi = buf.find(b"\xff\xd8")
        if soi == -1:
            return None, buf[-3:]
        if soi > 0:
            del buf[:soi]
        eoi = buf.find(b"\xff\xd9", 2)
        if eoi == -1:
            return None, buf
        return bytes(buf[: eoi + 2]), bytearray(buf[eoi + 2:])

    def _read_frame(self) -> Optional[bytes]:
        buf = self._jpeg_remainder
        while self._capture_running:
            frame, buf = self._extract_frame(buf)
            if frame is not None:
                self._jpeg_remainder = buf
                return frame
            try:
                data = self._pipe.read(65536)
            except Exception:
                data = b""
            if not data:
                break
            buf.extend(data)
            if len(buf) > MAX_FRAME_SIZE:
                buf = buf[-3:]
        self._jpeg_remainder = bytearray()
        return None

    def release_camera(self) -> None:
        self._stop_ffmpeg()
        if self.cap:
            self.cap.release()
            self.cap = None

    def _process_frame(self, raw: np.ndarray, left: bool, size=None) -> np.ndarray:
        h, w = raw.shape[:2]
        half = w // 2
        side = raw[:, :half] if left else raw[:, half:]
        h, w = side.shape[:2]
        target = size if size is not None else self.img_size
        if (w, h) != target:
            side = cv2.resize(side, target)
        return side

    def _decode_latest(self) -> Optional[np.ndarray]:
        with self.lock:
            raw = self._raw_jpeg
        if not raw:
            return None
        try:
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        return img

    def get_rectified_pair(self, size=None):
        """Оба глаза из одного декодированного stereo-кадра.

        Один decode на пару вместо двух (сбор данных вызывает это 10 раз/с),
        и оба глаза гарантированно из одного кадра.
        """
        if self.backend == "ffmpeg":
            raw = self._decode_latest()
            if raw is None:
                return None, None
        else:
            if not self.cap or not self.cap.isOpened():
                if not self.initialize_camera():
                    return None, None
            ret, raw = self.cap.read()
            if not ret:
                return None, None
        return (self._process_frame(raw, True, size),
                self._process_frame(raw, False, size))

    def update_params(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if value is not None and hasattr(self, key):
                    setattr(self, key, value)

    def _capture_loop(self) -> None:
        while self._capture_running:
            try:
                if self._sim_active:
                    self._capture_loop_sim()
                elif self.backend == "ffmpeg":
                    self._capture_loop_ffmpeg()
                else:
                    self._capture_loop_opencv()
            except Exception:
                time.sleep(0.01)

    def _capture_loop_ffmpeg(self) -> None:
        if self._ffmpeg_proc is None:
            if not self._start_ffmpeg():
                self._ffmpeg_failures += 1
                if self._ffmpeg_failures >= 3:
                    print("[Camera] Giving up on ffmpeg, switching to OpenCV backend", flush=True)
                    self.backend = "opencv"
                    if not self._init_opencv():
                        time.sleep(0.5)
                else:
                    time.sleep(1.0)
            return

        raw = self._read_frame()
        if raw is None:
            self._stop_ffmpeg()
            return

        self._publish_raw(raw)

    def _capture_loop_sim(self) -> None:
        time.sleep(1.0 / max(1, self.fps_target))
        self._publish_raw(self._sim_jpeg())

    def _sim_jpeg(self) -> bytes:
        """Синтетический stereo-кадр: надпись SIMULATION в каждом глазу."""
        w, h = self.capture_width, self.capture_height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        self._sim_tick += 1
        for cx in (w // 4, 3 * w // 4):
            cv2.putText(frame, "SIMULATION", (cx - 250, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6)
            cv2.putText(frame, f"frame {self._sim_tick}", (cx - 150, h // 2 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (170, 170, 170), 3)
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return jpeg.tobytes() if ok else b""

    def _publish_raw(self, raw: bytes) -> None:
        """Опубликовать полный stereo-JPEG (VR и декод-пути).

        Только байты, без декода: поток чтения камеры не должен
        затягиваться тяжёлой работой, иначе пайп ffmpeg переполняется.
        """
        with self.lock:
            self._raw_jpeg = raw
            self._latest_stereo_jpeg = raw
            self._stereo_seq += 1
            self._last_frame_mono = time.monotonic()
            self._frame_count += 1
            now = time.time()
            if now - self._last_frame_time >= 1.0:
                self.fps = self._frame_count / (now - self._last_frame_time)
                self._frame_count = 0
                self._last_frame_time = now

    def _eye_publisher_loop(self) -> None:
        """Глаз для /video_feed кодируется отдельным потоком, не в потоке чтения.

        Декод 2560x720 + resize + encode занимают десятки мс; если делать это
        в потоке чтения, пайп ffmpeg переполняется и задержка стрима растёт
        со временем. Здесь кодируем всегда СВЕЖИЙ кадр: не успеваем за
        камерой — пропускаем кадры, но задержка остаётся постоянной.
        """
        last_seq = -1
        while self._capture_running:
            try:
                with self.lock:
                    raw = self._raw_jpeg
                    seq = self._stereo_seq
                    left = self.show_left
                    have_consumer = self._feed_consumers > 0
                # глаз из raw-JPEG есть только у ffmpeg/sim-путей; opencv-путь
                # кодирует глаз сам в своём лупе.
                if not have_consumer or raw is None or self.backend != "ffmpeg":
                    last_seq = -1
                    time.sleep(0.05)
                    continue
                if seq == last_seq:
                    time.sleep(0.005)
                    continue
                last_seq = seq
                self._encode_eye(raw, left)
            except Exception:
                time.sleep(0.01)

    def _encode_eye(self, raw: bytes, left: bool) -> None:
        """Декод stereo-кадра -> кроп выбранного глаза -> JPEG для /video_feed."""
        try:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return
            if img.shape[1] != self.actual_width or img.shape[0] != self.actual_height:
                self.actual_width = img.shape[1]
                self.actual_height = img.shape[0]
                print(f"[Camera] delivered frame {self.actual_width}x{self.actual_height} "
                      f"(requested {self.capture_width}x{self.capture_height}), eye {self.eye_w}x{self.eye_h}",
                      flush=True)
            frame = self._process_frame(img, left)
            jpeg = cv2.imencode(".jpg", frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1].tobytes()
            with self.lock:
                self._latest_jpeg = jpeg
                self._jpeg_seq += 1
        except Exception:
            pass

    def _capture_loop_opencv(self) -> None:
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                time.sleep(0.5)
                return

        ret, raw = self.cap.read()
        if not ret:
            self.cap.release()
            self.cap = None
            time.sleep(0.1)
            return

        with self.lock:
            left = self.show_left
            need_eye = self._feed_consumers > 0
            need_stereo = self._stereo_consumers > 0
        frame = self._process_frame(raw, left)

        jpeg = None
        stereo_jpeg = None
        try:
            if need_eye:
                jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1].tobytes()
            if need_stereo:
                stereo_jpeg = cv2.imencode(".jpg", raw, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1].tobytes()
        except Exception:
            jpeg = None
            stereo_jpeg = None

        with self.lock:
            self._latest_frame = frame
            self._last_frame_mono = time.monotonic()
            if jpeg is not None:
                self._latest_jpeg = jpeg
                self._jpeg_seq += 1
            if stereo_jpeg is not None:
                self._latest_stereo_jpeg = stereo_jpeg
                self._stereo_seq += 1
            self._frame_count += 1
            now = time.time()
            if now - self._last_frame_time >= 1.0:
                self.fps = self._frame_count / (now - self._last_frame_time)
                self._frame_count = 0
                self._last_frame_time = now

    def start_continuous_capture(self) -> None:
        if self._capture_running:
            return
        self._capture_running = True
        self._last_frame_mono = time.monotonic()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-capture-thread")
        self._capture_thread.start()
        self._eye_thread = threading.Thread(target=self._eye_publisher_loop, daemon=True, name="camera-eye-thread")
        self._eye_thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="camera-watchdog")
        self._watchdog_thread.start()

    def stop_continuous_capture(self) -> None:
        self._capture_running = False
        self._stop_ffmpeg()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None
        if self._eye_thread and self._eye_thread.is_alive():
            self._eye_thread.join(timeout=1.0)
        self._eye_thread = None
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=WATCHDOG_PERIOD + 0.5)
        self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        """Если камера «отвалилась» — переподнять захват и USB-устройство."""
        while self._capture_running:
            time.sleep(WATCHDOG_PERIOD)
            if not self._capture_running:
                break
            with self.lock:
                stalled = time.monotonic() - self._last_frame_mono
                recovering = self._recovering
            if stalled < STALL_SEC or recovering:
                continue
            self._recovering = True
            try:
                print(f"[Camera] no frames for {stalled:.0f}s — recovering", flush=True)
                # Сначала пробуем просто переинициализировать захват.
                self.release_camera()
                self._ffmpeg_failures = 0
                self.backend = self._preferred_backend
                self._stop_ffmpeg()
                # Если устройство пропало с шины — программный «реплаг».
                try:
                    from robov_core import usb_reset
                    if not self._device_present():
                        usb_reset.reset(self.camera_source)
                except Exception as exc:
                    print(f"[Camera] usb_reset error: {exc}", flush=True)
                # Дать захвату заново подняться и сбросить счётчик простоя.
                with self.lock:
                    self._last_frame_mono = time.monotonic()
            finally:
                self._recovering = False

    def _device_present(self) -> bool:
        src = self.camera_source
        if isinstance(src, str):
            return os.path.exists(src)
        return os.path.exists(f"/dev/video{int(src)}")

    def get_latest_frame(self) -> np.ndarray:
        if self.backend == "ffmpeg":
            raw = self._decode_latest()
            if raw is not None:
                return raw
        with self.lock:
            if self._latest_frame is not None:
                return self._latest_frame
        frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def get_latest_jpeg(self) -> Tuple[Optional[bytes], int]:
        with self.lock:
            return self._latest_jpeg, self._jpeg_seq

    def add_consumer(self, stereo: bool = False) -> None:
        with self.lock:
            if stereo:
                self._stereo_consumers += 1
            else:
                self._feed_consumers += 1

    def remove_consumer(self, stereo: bool = False) -> None:
        with self.lock:
            if stereo:
                self._stereo_consumers = max(0, self._stereo_consumers - 1)
            else:
                self._feed_consumers = max(0, self._feed_consumers - 1)

    def get_stereo_jpeg(self) -> Tuple[Optional[bytes], int]:
        """Полный стерео-кадр (оба глаза) для VR-семплинга по UV."""
        with self.lock:
            return self._latest_stereo_jpeg, self._stereo_seq

    def layout(self) -> dict:
        """Раскладка глаз для клиента (UV-регионы левого/правого глаза).

        По умолчанию — простой разрез кадра по середине. Ручной оверврайд
        (eye_layout в cam_params.json) переопределяет значения.
        """
        with self.lock:
            if self._eye_layout_override:
                x0, x1, W = self._eye_x0, self._eye_x1, self._eye_crop_w
            else:
                half = self.actual_width // 2
                x0, x1, W = 0, half, half
            return {
                "x0": x0,
                "x1": x1,
                "w": W,
                "frame_w": self.actual_width,
                "frame_h": self.actual_height,
                "fov_h": self.cam_fov_h,
                "calibrated": True,
            }

    def debug_frame(self) -> Optional[bytes]:
        """JPEG с нарисованными кропами глаз — визуальная диагностика.

        Открыть в браузере, чтобы увидеть, что камера считает левым/правым
        глазом (зелёные рамки + подписи x=...).
        """
        if self.backend == "ffmpeg":
            raw = self._decode_latest()
        else:
            if not self.cap or not self.cap.isOpened():
                if not self.initialize_camera():
                    return None
            ret, raw = self.cap.read()
            if not ret:
                return None
        if raw is None:
            return None
        color = raw.copy() if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        h, w = color.shape[:2]
        if self._eye_layout_override:
            x0, x1, W = self._eye_x0, self._eye_x1, self._eye_crop_w
        else:
            half = w // 2
            x0, x1, W = 0, half, half
        green = (0, 200, 0)
        for xx, tag in ((x0, "L"), (x1, "R")):
            cv2.rectangle(color, (xx, 0), (xx + W, h), green, 2)
            cv2.putText(color, f"{tag} x={xx} w={W}",
                        (xx + 4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, green, 2)
        return cv2.imencode(".jpg", color,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85])[1].tobytes()
