import json
import os
import platform
import threading
import time
from typing import Optional, Tuple

import numpy as np

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2


class CameraInitError(Exception):
    pass


class StereoCamera:
    def __init__(
        self,
        camera_param_file: str = "cam_params.json",
        source: int = 0,
        capture_width: int = 1280,
        capture_height: int = 720,
    ):
        self.camera_param_file: str = camera_param_file
        self.camera_source: int = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock: threading.Lock = threading.Lock()

        self.img_size: Tuple[int, int] = (640, 360)
        self.show_left: bool = True
        self.fps: float = 30.0
        self._tick: int = 0
        self._last_frame_time: float = time.time()
        self._frame_count: int = 0

        self.capture_width: int = capture_width
        self.capture_height: int = capture_height
        self.actual_width: int = 0
        self.actual_height: int = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running: bool = False

        try:
            with open(self.camera_param_file, encoding="utf-8") as f:
                params = json.load(f)
            self.capture_width = int(params.get("capture_width", self.capture_width))
            self.capture_height = int(params.get("capture_height", self.capture_height))
            isize = params.get("img_size")
            if isinstance(isize, (list, tuple)) and len(isize) >= 2:
                self.img_size = (int(isize[0]), int(isize[1]))
        except Exception:
            pass

    def initialize_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                return False
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
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

    def release_camera(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def _process_frame(self, raw: np.ndarray, left: bool) -> np.ndarray:
        half_w = raw.shape[1] // 2
        side = raw[:, :half_w] if left else raw[:, half_w:]
        h, w = side.shape[:2]
        if (w, h) != self.img_size:
            side = cv2.resize(side, self.img_size)
        return side

    def get_rectified_frame(self, left: bool = True) -> Optional[np.ndarray]:
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                return None
        ret, raw = self.cap.read()
        if not ret:
            return None
        return self._process_frame(raw, left)

    def get_frame(self) -> np.ndarray:
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA ERROR", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                return frame
        ret, raw = self.cap.read()
        if not ret:
            self.cap.release()
            self.cap = None
            frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA ERROR", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            return frame

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
                if not self.cap or not self.cap.isOpened():
                    if not self.initialize_camera():
                        time.sleep(0.5)
                        continue

                ret, raw = self.cap.read()
                if not ret:
                    self.cap.release()
                    self.cap = None
                    time.sleep(0.1)
                    continue

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
                    self._frame_count += 1
                    now = time.time()
                    if now - self._last_frame_time >= 1.0:
                        self.fps = self._frame_count / (now - self._last_frame_time)
                        self._frame_count = 0
                        self._last_frame_time = now
            except Exception:
                time.sleep(0.01)

    def start_continuous_capture(self) -> None:
        if self._capture_running:
            return
        self._capture_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="camera-capture-thread")
        self._capture_thread.start()

    def stop_continuous_capture(self) -> None:
        self._capture_running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None

    def get_latest_frame(self) -> np.ndarray:
        with self.lock:
            if self._latest_frame is not None:
                return self._latest_frame
        frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self._latest_jpeg
