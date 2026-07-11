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
        self.depth_enabled: bool = False
        self.hud_enabled: bool = False
        self.alpha_depth: float = 0.3
        self.show_left: bool = True
        self.wls_enabled: bool = True
        self.fps: float = 30.0
        self._tick: int = 0
        self._last_frame_time: float = time.time()
        self._frame_count: int = 0

        self.camera_fov: int = 120
        self.window_size: int = 11
        self.min_disp: int = 0
        self.num_disp: int = 256

        self.actual_width: int = 0
        self.actual_height: int = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running: bool = False

        self._load_camera_parameters()
        self._setup_rectification()
        self.focal_length = self.P1[0, 0]
        self.baseline = abs(float(self.T[0]))
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
            if self.cap.isOpened():
                self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return True
            return False
        except Exception as e:
            raise CameraInitError(f"Failed to open camera: {e}") from e

    def release_camera(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_rectified_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
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

        displ = self.left_matcher.compute(grayL, grayR)

        if self.wls_enabled and self.right_matcher and self.wls_filter:
            dispr = self.right_matcher.compute(grayR, grayL)
            filtered_disp = self.wls_filter.filter(displ, grayL, disparity_map_right=dispr)
        else:
            filtered_disp = displ

        filtered_disp[filtered_disp < 0] = 0
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

        ret, raw = self.cap.read()
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
        if len(self.disp_buffer) > 0:
            disp = self.disp_buffer[-1]
            sh, sw = disp.shape[:2]
            sx = int(x * sw / self.imSize[0])
            sy = int(y * sh / self.imSize[1])
            sx = max(0, min(sx, sw - 1))
            sy = max(0, min(sy, sh - 1))
            return self.get_depth_at_point(disp, sx, sy)
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

            w, h = self.imSize
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
            matcher_keys = {"wls_enabled", "num_disp", "window_size", "min_disp"}
            if matcher_keys & kwargs.keys():
                self._setup_stereo_matchers()

    # --- Shared frame buffer ---

    def _process_raw_frame(self, raw: np.ndarray) -> np.ndarray:
        raw = cv2.rotate(raw, cv2.ROTATE_180)
        half_w = raw.shape[1] // 2

        if self.depth_enabled or self.hud_enabled:
            imgL = cv2.remap(raw[:, :half_w], self.lMapX, self.lMapY, cv2.INTER_LINEAR)
            imgR = cv2.remap(raw[:, half_w:], self.rMapX, self.rMapY, cv2.INTER_LINEAR)
            frame = imgL if self.show_left else imgR
            frame = cv2.resize(frame, self.img_size)

            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            grayL_d = cv2.resize(grayL, self.img_size, interpolation=cv2.INTER_LINEAR)
            grayR_d = cv2.resize(grayR, self.img_size, interpolation=cv2.INTER_LINEAR)

            displ = self.left_matcher.compute(grayL_d, grayR_d)
            displ[displ < 0] = 0

            wls_active = self.wls_enabled and self.right_matcher and self.wls_filter
            disp_final = displ
            if wls_active:
                dispr = self.right_matcher.compute(grayR_d, grayL_d)
                disp_final = self.wls_filter.filter(displ, grayL_d, disparity_map_right=dispr)
                disp_final[disp_final < 0] = 0

            self.disp_buffer.append(disp_final.copy())

            if self.depth_enabled:
                depth_vis = cv2.normalize(disp_final, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_vis = cv2.morphologyEx(depth_vis, cv2.MORPH_OPEN, self.kernel)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.addWeighted(frame, 1 - self.alpha_depth, depth_vis, self.alpha_depth, 0, dst=frame)

            if self.hud_enabled:
                frame = self._render_hud(frame, disp_final, center_only=False)
        else:
            rawL = raw[:, :half_w]
            rawR = raw[:, half_w:]
            side = rawL if self.show_left else rawR
            frame = cv2.resize(side, self.img_size)

        return frame

    def _render_hud(self, frame: np.ndarray, disp_raw: np.ndarray, center_only: bool = False) -> np.ndarray:
        h, w = frame.shape[:2]
        dh, dw = disp_raw.shape[:2]
        result = frame.copy()

        f_eff = self.focal_length * dh / self.imSize[1]
        d16 = disp_raw.astype(np.float32) / 16.0

        with np.errstate(divide='ignore'):
            Z = np.where(d16 > 1.0, f_eff * self.baseline / d16, 0.0)
        Z = np.clip(Z, 0, 50000)

        fx = dw // 2
        fy = dh // 2

        center_z = float(Z[fy, fx])

        if center_only:
            nearest_z = center_z
            nearest_idx = h // 2
        else:
            box_hw = dw // 8
            box_hh = dh // 8
            x1 = max(0, fx - box_hw)
            x2 = min(dw, fx + box_hw)
            y1 = max(0, fy - box_hh)
            y2 = min(dh, fy + box_hh)

            region = Z[y1:y2, x1:x2]
            valid = region > 0
            if np.any(valid):
                nearest_z = float(np.min(region[valid]))
                min_idx = np.unravel_index(np.argmin(np.where(valid, region, np.inf)), region.shape)
                nearest_idx = int((y1 + min_idx[0]) * h / dh)
            else:
                nearest_z = 0.0
                nearest_idx = -1

        font = cv2.FONT_HERSHEY_SIMPLEX

        def draw_text_box(img, lines, x, y, font_scale=0.45, pad=6):
            line_h = 18
            max_w = max(cv2.getTextSize(l, font, font_scale, 1)[0][0] for l in lines)
            box_h = len(lines) * line_h + pad * 2
            x1 = x
            y1 = y
            x2 = x1 + max_w + pad * 2
            y2 = y1 + box_h
            cv2.rectangle(img, (x1, y1), (x2, y2), (10, 10, 10), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (42, 42, 42), 1)
            for i, line in enumerate(lines):
                ty = y1 + pad + i * line_h + 12
                cv2.putText(img, line, (x1 + pad, ty), font, font_scale, (240, 240, 240), 1)

        def fmt(d):
            if d <= 0:
                return "--"
            if d >= 10000:
                return f"{d/1000:.1f} m"
            return f"{d:.0f} mm"

        draw_text_box(result, [f"Center: {fmt(center_z)}", f"Near: {fmt(nearest_z)}"], 12, h - 56)

        overlay = result.copy()
        half_fov = self.camera_fov // 2
        for offset in range(-half_fov, half_fov + 1, 20):
            x = int((offset + half_fov) / self.camera_fov * w)
            text = "0" if offset == 0 else str(offset)
            cv2.putText(overlay, text, (x - 6, 16), font, 0.4, (0, 255, 0), 1)
            cv2.line(overlay, (x, 24), (x, h), (0, 255, 0), 1)

        for i in range(1, 4):
            y = int(h * i / 4)
            cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 1)

        cv2.addWeighted(overlay, 0.35, result, 0.65, 0, dst=result)

        if nearest_z > 0 and nearest_idx >= 0:
            cv2.circle(result, (w // 2, nearest_idx), 4, (0, 255, 255), -1)

        return result

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
