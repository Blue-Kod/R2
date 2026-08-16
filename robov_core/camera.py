import json
import os
import platform
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

MAX_FRAME_SIZE = 4 * 1024 * 1024


class CameraInitError(Exception):
    pass


class StereoCamera:
    def __init__(
        self,
        camera_param_file: str = "cam_params.json",
        source: int = 0,
        capture_width: int = 2560,
        capture_height: int = 720,
        backend: str = "ffmpeg",
        fps_target: int = 30,
    ):
        self.camera_param_file: str = camera_param_file
        self.camera_source: int = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock: threading.Lock = threading.Lock()

        self.img_size: Tuple[int, int] = (640, 360)
        self.eye_w: int = 1280
        self.eye_h: int = 720
        self.show_left: bool = True
        self.fps: float = 30.0
        self._tick: int = 0
        self._last_frame_time: float = time.time()
        self._frame_count: int = 0

        self.capture_width: int = capture_width
        self.capture_height: int = capture_height
        self.backend: str = backend
        self.fps_target: int = fps_target
        self.actual_width: int = 0
        self.actual_height: int = 0
        self._eye_x0: int = 0
        self._eye_x1: int = self.eye_w
        self._eye_crop_w: int = self.eye_w
        self._layout_checked: bool = False
        self._layout_attempts: int = 0
        self._best_layout: Optional[Tuple[float, int, int, int]] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_seq: int = 0
        self._raw_jpeg: Optional[bytes] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running: bool = False

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
            self.fps_target = int(params.get("fps_target", self.fps_target))
            isize = params.get("img_size")
            if isinstance(isize, (list, tuple)) and len(isize) >= 2:
                self.img_size = (int(isize[0]), int(isize[1]))
            im_size = params.get("imSize")
            if isinstance(im_size, (list, tuple)) and len(im_size) >= 2:
                self.eye_w = int(im_size[0])
                self.eye_h = int(im_size[1])
        except Exception:
            pass

    def initialize_camera(self) -> bool:
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
        return self._init_opencv()

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
        dev = f"/dev/video{self.camera_source}"
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

    def _calibrate_layout(self, raw: np.ndarray) -> bool:
        """Автопоиск раскладки стерео-кадра по одному кадру.

        Ищем (x0_левого, ширина, x0_правого), где половинки кадра максимально
        похожи (NCC колонных профилей яркости — стерео-пара почти идентична).
        Покрывает кадр шире 2560, зазор между глазами и правый глаз не на
        [1280:2560]. Возвращает True, если найдена уверенная пара (ncc >= 0.6).
        """
        h, w = raw.shape[:2]
        if w < 2 * 600:
            return False
        try:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
        except cv2.error:
            return False
        col = gray.mean(axis=0)
        col -= col.mean()
        norm_all = float(np.sqrt(float(np.sum(col * col)))) + 1e-9
        if norm_all <= 1e-6:
            return False
        col /= norm_all
        pref = np.zeros(w + 1, dtype=np.float64)
        np.cumsum(col * col, out=pref[1:])

        W_lo = max(600, int(self.eye_w * 0.6))
        W_hi = min(w // 2, int(self.eye_w * 1.4))
        A_hi = min(256, w // 8)
        best = None
        for A in range(0, A_hi + 1, 16):
            for W in range(W_lo, W_hi + 1, 16):
                if A + W > w // 2:
                    continue
                left = col[A:A + W]
                n_left = float(np.sqrt(float(np.sum(left * left)))) + 1e-9
                cc = np.correlate(col, left, mode="valid")
                sq = pref[W:] - pref[:-W]
                ncc = cc / (np.sqrt(sq) * n_left + 1e-9)
                min_x1 = A + int(W * 0.8)
                if min_x1 < len(ncc):
                    ncc[:min_x1] = -np.inf
                i = int(np.argmax(ncc))
                # Tie-break: при близких ncc предпочитаем полный глаз
                # (W ~ eye_w) и левый глаз от края кадра (A ~ 0).
                score = ncc[i] + 0.001 * (W / self.eye_w) \
                        + 0.0005 * (1.0 - A / max(A_hi, 1))
                if best is None or score > best[0]:
                    best = (float(score), float(ncc[i]), A, W, i)
        if best is None:
            return False
        score, ncc, x0, W, x1 = best
        self._best_layout = best
        if ncc >= 0.6 or self._layout_attempts >= 5:
            self._eye_x0 = x0
            self._eye_x1 = x1
            self._eye_crop_w = W
            self._layout_checked = True
            print(f"[Camera] layout: left x={x0}, right x={x1}, "
                  f"eye w={W}, ncc={ncc:.3f} (frame {self.actual_width}x{self.actual_height})",
                  flush=True)
            return ncc >= 0.6
        return False

    def _maybe_calibrate(self, raw: np.ndarray) -> None:
        if self._layout_checked:
            return
        self._layout_attempts += 1
        self._calibrate_layout(raw)

    def _process_frame(self, raw: np.ndarray, left: bool) -> np.ndarray:
        self._maybe_calibrate(raw)
        h, w = raw.shape[:2]
        eye_w = self.eye_w
        x0 = self._eye_x0 if left else self._eye_x1
        W = self._eye_crop_w
        if x0 + W <= w:
            side = raw[:, x0:x0 + W]
        elif w >= 2 * eye_w:
            x0 = 0 if left else eye_w
            side = raw[:, x0:x0 + eye_w]
        else:
            half_w = w // 2
            side = raw[:, :half_w] if left else raw[:, half_w:]
        h, w = side.shape[:2]
        if (w, h) != self.img_size:
            side = cv2.resize(side, self.img_size)
        return side

    def _no_camera_frame(self) -> np.ndarray:
        frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
        cv2.putText(frame, "CAMERA ERROR", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        return frame

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

    def get_rectified_frame(self, left: bool = True) -> Optional[np.ndarray]:
        if self.backend == "ffmpeg":
            raw = self._decode_latest()
            if raw is None:
                return None
            return self._process_frame(raw, left)
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                return None
        ret, raw = self.cap.read()
        if not ret:
            return None
        return self._process_frame(raw, left)

    def get_frame(self) -> np.ndarray:
        if self.backend == "ffmpeg":
            raw = self._decode_latest()
            if raw is None:
                return self._no_camera_frame()
            with self.lock:
                left = self.show_left
            return self._process_frame(raw, left)
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                return self._no_camera_frame()
        ret, raw = self.cap.read()
        if not ret:
            self.cap.release()
            self.cap = None
            return self._no_camera_frame()

        with self.lock:
            left = self.show_left
        frame = self._process_frame(raw, left)

        with self.lock:
            self._frame_count += 1
            now = time.time()
            if now - self._last_frame_time >= 1.0:
                self.fps = self._frame_count / (now - self._last_frame_time)
                self._frame_count = 0
                self._last_frame_time = now

        return frame

    def update_params(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if value is not None and hasattr(self, key):
                    setattr(self, key, value)

    def _capture_loop(self) -> None:
        while self._capture_running:
            try:
                if self.backend == "ffmpeg":
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

        with self.lock:
            self._raw_jpeg = raw
            left = self.show_left
            self._jpeg_seq += 1
            self._frame_count += 1
            now = time.time()
            if now - self._last_frame_time >= 1.0:
                self.fps = self._frame_count / (now - self._last_frame_time)
                self._frame_count = 0
                self._last_frame_time = now

        # Стрим/бразузер/VR должны видеть один глаз (левый/правый), а не
        # склейку stereo-кадра. raw_jpeg хранит полный кадр для декод-путей,
        # а latest_jpeg — обрезанную до выбранного глаза JPEG (как в opencv).
        try:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
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
        frame = self._process_frame(raw, left)

        try:
            jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])[1].tobytes()
        except Exception:
            jpeg = None

        with self.lock:
            self._latest_frame = frame
            if jpeg is not None:
                self._latest_jpeg = jpeg
                self._jpeg_seq += 1
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
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-capture-thread")
        self._capture_thread.start()

    def stop_continuous_capture(self) -> None:
        self._capture_running = False
        self._stop_ffmpeg()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None

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
