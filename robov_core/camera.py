import os
import threading
import time
import json
from collections import deque
from typing import Optional, Tuple

import numpy as np

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2


class CameraInitError(Exception):
    pass


class CameraReadError(Exception):
    pass


class StereoCamera:
    def __init__(self, camera_param_file: str = "cam_params.json", source: int = 0):
        self.camera_param_file: str = camera_param_file
        self.camera_source: int = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.disp_buffer: deque = deque(maxlen=5)
        self.lock: threading.Lock = threading.Lock()

        self.img_size: Tuple[int, int] = (640, 360)
        self.low_size: Tuple[int, int] = (160, 120)
        self.depth_enabled: bool = False
        self.alpha_depth: float = 0.3
        self.show_left: bool = True
        self.num_disp: int = 5
        self.wls_enabled: bool = True
        self.fps: float = 30.0
        self._tick: int = 0
        self._last_frame_time: float = time.time()
        self._frame_count: int = 0

        self.window_size: int = 11
        self.min_disp: int = 0
        self.num_disp: int = 128
        self._remap_flags: int = cv2.INTER_NEAREST

        self._latest_frame: Optional[np.ndarray] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running: bool = False

        self._load_camera_parameters()
        self._setup_rectification()
        self._setup_stereo_matchers()

    def _load_camera_parameters(self) -> None:
        try:
            with open(self.camera_param_file) as fp:
                cp = json.load(fp)
                self.Kl: np.ndarray = np.array(cp["Kl"])
                self.Dl: np.ndarray = np.array(cp["Dl"])
                self.Kr: np.ndarray = np.array(cp["Kr"])
                self.Dr: np.ndarray = np.array(cp["Dr"])
                self.R: np.ndarray = np.array(cp["R"])
                self.T: np.ndarray = np.array(cp["T"])
                self.imSize: Tuple[int, int] = tuple(cp["imSize"])
        except Exception as e:
            raise CameraInitError(f"Failed to load camera parameters: {e}") from e

    def _setup_rectification(self) -> None:
        self.R1, self.R2, self.P1, self.P2, self.Q = cv2.fisheye.stereoRectify(
            self.Kl, self.Dl, self.Kr, self.Dr, self.imSize, self.R, self.T,
            flags=cv2.fisheye.CALIB_ZERO_DISPARITY, balance=0.0
        )
        self.lMapX, self.lMapY = cv2.fisheye.initUndistortRectifyMap(
            self.Kl, self.Dl, self.R1, self.P1, self.imSize, cv2.CV_32FC1
        )
        self.rMapX, self.rMapY = cv2.fisheye.initUndistortRectifyMap(
            self.Kr, self.Dr, self.R2, self.P2, self.imSize, cv2.CV_32FC1
        )

    def _setup_stereo_matchers(self) -> None:
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.window_size,
            P1=8 * 3 * self.window_size ** 2,
            P2=32 * 3 * self.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=200,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        if self.depth_enabled or self.wls_enabled:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
            self.wls_filter.setLambda(8000.0)
            self.wls_filter.setSigmaColor(1.5)
        else:
            self.right_matcher = None
            self.wls_filter = None

        self.kernel = np.ones((3, 3), np.uint8)

    def initialize_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return self.cap.isOpened()
        except Exception as e:
            raise CameraInitError(f"Failed to open camera: {e}") from e

    def release_camera(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_rectified_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                return None, None

        ret, frame = self.cap.read()
        if not ret:
            if not self.initialize_camera():
                return None, None
            ret, frame = self.cap.read()
            if not ret:
                return None, None

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        half_w = frame.shape[1] // 2
        imgL = cv2.remap(frame[:, :half_w], self.lMapX, self.lMapY, cv2.INTER_LINEAR)
        imgR = cv2.remap(frame[:, half_w:], self.rMapX, self.rMapY, cv2.INTER_LINEAR)

        return imgL, imgR

    def compute_disparity(self, left_frame: np.ndarray, right_frame: np.ndarray) -> np.ndarray:
        grayL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        orig_h, orig_w = grayL.shape

        if not self.depth_enabled:
            grayL = cv2.resize(grayL, (0, 0), fx=0.5, fy=0.5)
            grayR = cv2.resize(grayR, (0, 0), fx=0.5, fy=0.5)

        displ = self.left_matcher.compute(grayL, grayR)

        if (self.depth_enabled or self.wls_enabled) and self.right_matcher and self.wls_filter:
            dispr = self.right_matcher.compute(grayR, grayL)
            filtered_disp = self.wls_filter.filter(displ, grayL, disparity_map_right=dispr)
        else:
            filtered_disp = displ

        filtered_disp[filtered_disp < 0] = 0

        if not self.depth_enabled:
            filtered_disp = cv2.resize(filtered_disp, (orig_w, orig_h))

        if self.depth_enabled:
            self.disp_buffer.append(filtered_disp.copy())

        return filtered_disp

    def get_depth_at_point(self, disparity_map: np.ndarray, x: Optional[int] = None, y: Optional[int] = None) -> float:
        points_3d = cv2.reprojectImageTo3D(disparity_map.astype(np.float32) / 16.0, self.Q)
        h, w = disparity_map.shape[:2]
        if x is None:
            x = w // 2
        if y is None:
            y = h // 2
        return float(abs(points_3d[y, x][2]))

    def get_average_depth_map(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if len(self.disp_buffer) == 0:
            return None, None
        avg_disp = np.mean(self.disp_buffer, axis=0).astype(np.int16)
        center_depth = self.get_depth_at_point(avg_disp)
        return avg_disp, center_depth

    def visualize_disparity(self, disparity_map: np.ndarray) -> np.ndarray:
        disp_vis = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_vis = cv2.morphologyEx(disp_vis, cv2.MORPH_OPEN, self.kernel)
        return cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    def add_depth_text(self, frame: np.ndarray, depth_mm: float, position: Tuple[int, int] = (30, 50)) -> np.ndarray:
        txt = f"Depth: {depth_mm:.1f} mm" if depth_mm < 5000 else "Out of range"
        result = frame.copy()
        cv2.putText(result, txt, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return result

    def capture_frame_with_depth(self) -> Optional[dict]:
        left_frame, right_frame = self.get_rectified_frames()
        if left_frame is None:
            return None
        disparity_map = self.compute_disparity(left_frame, right_frame)
        depth_mm = self.get_depth_at_point(disparity_map)
        return {
            'left_frame': left_frame,
            'right_frame': right_frame,
            'disparity_map': disparity_map,
            'depth_mm': depth_mm
        }

    def clear_buffer(self) -> None:
        self.disp_buffer.clear()

    def get_frame(self) -> np.ndarray:
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA ERROR", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                return frame

        for _ in range(3):
            if not self.cap.grab():
                break
        ret, raw = self.cap.retrieve()
        if not ret:
            self.cap.release()
            self.cap = None
            frame = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA ERROR", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            return frame

        frame = self._process_raw_frame(raw)

        with self.lock:
            self._frame_count += 1
            now = time.time()
            if now - self._last_frame_time >= 1.0:
                self.fps = self._frame_count / (now - self._last_frame_time)
                self._frame_count = 0
                self._last_frame_time = now

        return frame

    def get_depth_at(self, x: int, y: int) -> float:
        left_frame, right_frame = self.get_rectified_frames()
        if left_frame is None:
            return 120.0
        disparity_map = self.compute_disparity(left_frame, right_frame)
        return self.get_depth_at_point(disparity_map, x, y)

    def get_real_coords(self, x_px: int, y_px: int) -> Optional[dict]:
        try:
            depth_mm = self.get_depth_at(x_px, y_px)
            if depth_mm is None or depth_mm <= 0:
                return None

            h, w = self.imSize
            if x_px < 0 or x_px >= w or y_px < 0 or y_px >= h:
                return None

            fx = self.Kl[0, 0]
            fy = self.Kl[1, 1]

            x_real = (x_px - self.Kl[0, 2]) * depth_mm / fx
            y_real = (y_px - self.Kl[1, 2]) * depth_mm / fy
            z_real = depth_mm

            return {'x': float(x_real) / 1000.0, 'y': float(y_real) / 1000.0, 'z': float(z_real) / 1000.0}

        except Exception:
            return None

    def update_params(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if value is not None and hasattr(self, key):
                    setattr(self, key, value)

    # --- Shared frame buffer ---

    def _process_raw_frame(self, raw: np.ndarray) -> np.ndarray:
        raw = cv2.rotate(raw, cv2.ROTATE_180)
        half_w = raw.shape[1] // 2
        imgL = cv2.remap(raw[:, :half_w], self.lMapX, self.lMapY, self._remap_flags)
        imgR = cv2.remap(raw[:, half_w:], self.rMapX, self.rMapY, self._remap_flags)

        frame = imgL if self.show_left else imgR
        frame = cv2.resize(frame, self.img_size)

        if self.depth_enabled:
            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            h, w = grayL.shape
            grayL = cv2.resize(grayL, (0, 0), fx=0.5, fy=0.5)
            grayR = cv2.resize(grayR, (0, 0), fx=0.5, fy=0.5)
            displ = self.left_matcher.compute(grayL, grayR)
            if self.wls_enabled and self.right_matcher and self.wls_filter:
                dispr = self.right_matcher.compute(grayR, grayL)
                filtered_disp = self.wls_filter.filter(displ, grayL, disparity_map_right=dispr)
            else:
                filtered_disp = displ
            filtered_disp[filtered_disp < 0] = 0
            filtered_disp = cv2.resize(filtered_disp, (w, h))

            depth_vis = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_vis = cv2.morphologyEx(depth_vis, cv2.MORPH_OPEN, self.kernel)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_resized = cv2.resize(depth_vis, self.img_size)
            cv2.addWeighted(frame, 1 - self.alpha_depth, depth_resized, self.alpha_depth, 0, dst=frame)

        return frame

    def _capture_loop(self) -> None:
        while self._capture_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    if not self.initialize_camera():
                        time.sleep(0.5)
                        continue

                for _ in range(3):
                    if not self.cap.grab():
                        break
                ret, raw = self.cap.retrieve()
                if not ret:
                    self.cap.release()
                    self.cap = None
                    time.sleep(0.1)
                    continue

                frame = self._process_raw_frame(raw)

                with self.lock:
                    self._latest_frame = frame
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
