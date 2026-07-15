import os
import platform
import threading
import time
import json
from collections import deque
from typing import Optional, Tuple

import numpy as np

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

from robov_core.depth_providers import DepthProvider, StereoSGBMDepthProvider, DepthResult
from robov_core.detector import ObjectDetector, Detection


class CameraInitError(Exception):
    pass


class CameraReadError(Exception):
    pass


class StereoCamera:
    def __init__(self, camera_param_file: str = "cam_params.json", source: int = 0):
        self.camera_param_file: str = camera_param_file
        self.camera_source: int = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.disp_buffer: deque = deque(maxlen=2)
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
        self.num_disp: int = 160
        self.depth_scale: float = 1.25

        self.detection_enabled: bool = False
        self.detector: ObjectDetector = ObjectDetector()
        self._last_detections: list = []
        self.detection_prompts: str = ""
        self._scan_cache: Optional[list] = None
        self._scan_cache_time: float = 0.0
        self._detection_frame_skip: int = 5
        self._detection_frame_count: int = 0

        self.actual_width: int = 0
        self.actual_height: int = 0
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_left: Optional[np.ndarray] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_running: bool = False

        self._load_camera_parameters()
        self._setup_rectification()
        self.focal_length = self.P1[0, 0]
        self.baseline = abs(float(self.T[0]))
        self._compute_scaled_intrinsics()
        self.depth_provider: DepthProvider = self._create_default_provider()
        self._init_provider()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
            flags=cv2.fisheye.CALIB_ZERO_DISPARITY, balance=0.8
        )
        self.lMapX, self.lMapY = cv2.fisheye.initUndistortRectifyMap(
            self.Kl, self.Dl, self.R1, self.P1, self.imSize, cv2.CV_32FC1
        )
        self.rMapX, self.rMapY = cv2.fisheye.initUndistortRectifyMap(
            self.Kr, self.Dr, self.R2, self.P2, self.imSize, cv2.CV_32FC1
        )

    def _compute_scaled_intrinsics(self) -> None:
        sx = self.img_size[0] / self.imSize[0]
        sy = self.img_size[1] / self.imSize[1]
        self._Kl_s = self.Kl.copy()
        self._Kl_s[0, 0] *= sx
        self._Kl_s[1, 1] *= sy
        self._Kl_s[0, 2] *= sx
        self._Kl_s[1, 2] *= sy
        self._Q_proc = self.Q.copy()
        self._Q_proc[0, 3] *= sx
        self._Q_proc[1, 3] *= sy
        self._Q_proc[2, 3] *= sx

    @staticmethod
    def _create_default_provider() -> DepthProvider:
        return StereoSGBMDepthProvider()

    def _init_provider(self) -> None:
        self.depth_provider.setup(
            num_disp=self.num_disp,
            window_size=self.window_size,
            min_disp=self.min_disp,
            wls_enabled=self.depth_enabled or self.wls_enabled,
        )
        self.kernel = np.ones((3, 3), np.uint8)

    def set_depth_provider(self, provider: DepthProvider) -> None:
        with self.lock:
            self.depth_provider.release()
            self.depth_provider = provider
            self._init_provider()
        self.disp_buffer.clear()

    def initialize_camera(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cap.isOpened():
                self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if platform.system() == "Linux":
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
                    except Exception:
                        pass
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
        grayL = cv2.resize(grayL, self.img_size)
        grayR = cv2.resize(grayR, self.img_size)
        result = self.depth_provider.compute(grayL, grayR)
        return result.disparity

    def get_depth_at_point(self, disparity_map: np.ndarray, x: Optional[int] = None, y: Optional[int] = None) -> float:
        points_3d = cv2.reprojectImageTo3D(disparity_map.astype(np.float32) / 16.0, self._Q_proc)
        h, w = disparity_map.shape[:2]
        if x is None:
            x = w // 2
        if y is None:
            y = h // 2
        return float(abs(points_3d[y, x][2])) * self.depth_scale

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

    def get_shared_depth(self):
        if len(self.disp_buffer) < 2:
            return None, None
        avg_disp = np.mean(list(self.disp_buffer), axis=0).astype(np.int16)
        with self.lock:
            left = self._latest_left
        return left, avg_disp

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
            sx = max(0, min(x, sw - 1))
            sy = max(0, min(y, sh - 1))
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

            w, h = self.img_size
            if x_px < 0 or x_px >= w or y_px < 0 or y_px >= h:
                return None

            fx = self._Kl_s[0, 0]
            fy = self._Kl_s[1, 1]

            x_real = (x_px - self._Kl_s[0, 2]) * depth_mm / fx
            y_real = (y_px - self._Kl_s[1, 2]) * depth_mm / fy
            z_real = depth_mm

            return {'x': float(x_real) / 1000.0, 'y': float(y_real) / 1000.0, 'z': float(z_real) / 1000.0, 'depth': float(depth_mm) / 1000.0}

        except Exception:
            return None

    def _get_mask_3d(self, mask: np.ndarray, erode_px: int = 3) -> Optional[dict]:
        if len(self.disp_buffer) == 0 or mask is None:
            return None
        disp = self.disp_buffer[-1]
        sh, sw = disp.shape[:2]
        if mask.shape[:2] != (sh, sw):
            mask = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((erode_px, erode_px), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        valid = (eroded > 0) & (disp > 0)
        n_valid = int(valid.sum())
        if n_valid < 5:
            return None
        points_3d = cv2.reprojectImageTo3D(
            disp.astype(np.float32) / 16.0, self._Q_proc
        )
        pts = points_3d[valid].reshape(-1, 3)
        depth_vals = np.abs(pts[:, 2]) * self.depth_scale
        range_ok = (depth_vals > 50.0) & (depth_vals < 5000.0)
        if range_ok.sum() < 3:
            return None
        pts = pts[range_ok]
        ds = self.depth_scale
        fx = self._Kl_s[0, 0]
        fy = self._Kl_s[1, 1]
        cx = self._Kl_s[0, 2]
        cy = self._Kl_s[1, 2]
        x_m = (pts[:, 0] * ds) / 1000.0
        y_m = (pts[:, 1] * ds) / 1000.0
        z_m = (np.abs(pts[:, 2]) * ds) / 1000.0
        mx, my, mz = float(x_m.mean()), float(y_m.mean()), float(z_m.mean())
        vx = float(x_m.max() - x_m.min())
        vy = float(y_m.max() - y_m.min())
        vz = float(z_m.max() - z_m.min())
        return {"x": mx, "y": my, "z": mz, "vx": vx, "vy": vy, "vz": vz, "n_points": n_valid}

    def update_params(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if value is not None and hasattr(self, key):
                    setattr(self, key, value)
            matcher_keys = {"wls_enabled", "num_disp", "window_size", "min_disp"}
            if matcher_keys & kwargs.keys():
                self._init_provider()

    # --- Shared frame buffer ---

    def find(self, name: str) -> Optional[dict]:
        left_frame, right_frame = self.get_rectified_frames()
        if left_frame is None:
            return None
        detections = self.detector.find(name, left_frame)
        if not detections:
            return None
        det = detections[0]
        result = {
            "name": det.name,
            "confidence": round(det.confidence, 3),
            "bbox": {"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2},
            "center": {"x": det.center_x, "y": det.center_y},
        }
        m3d = self._get_mask_3d(det.mask) if det.mask is not None else None
        if m3d:
            result["x"] = round(m3d["x"], 3)
            result["y"] = round(m3d["y"], 3)
            result["z"] = round(m3d["z"], 3)
            result["vx"] = round(m3d["vx"], 3)
            result["vy"] = round(m3d["vy"], 3)
            result["vz"] = round(m3d["vz"], 3)
            result["depth"] = round(m3d["z"], 3)
        else:
            coords = self.get_real_coords(det.center_x, det.center_y)
            if coords:
                result["x"] = round(coords["x"], 3)
                result["y"] = round(coords["y"], 3)
                result["z"] = round(coords["z"], 3)
                result["depth"] = round(coords["depth"], 3)
        return result

    def scan(self, prompts: Optional[str] = None, max_age: float = 2.0) -> list:
        now = time.time()
        if self._scan_cache is not None and (now - self._scan_cache_time) < max_age:
            return self._scan_cache
        left_frame, right_frame = self.get_rectified_frames()
        if left_frame is None:
            return []
        prompts = prompts or self.detection_prompts
        all_dets = self.detector.detect(left_frame)
        if prompts:
            all_dets = self._filter_by_prompts(all_dets, prompts)
        results = []
        for det in all_dets:
            item = {
                "name": det.name,
                "confidence": round(det.confidence, 3),
                "bbox": {"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2},
                "center": {"x": det.center_x, "y": det.center_y},
            }
            m3d = self._get_mask_3d(det.mask) if det.mask is not None else None
            if m3d:
                item["x"] = round(m3d["x"], 3)
                item["y"] = round(m3d["y"], 3)
                item["z"] = round(m3d["z"], 3)
                item["vx"] = round(m3d["vx"], 3)
                item["vy"] = round(m3d["vy"], 3)
                item["vz"] = round(m3d["vz"], 3)
                item["depth"] = round(m3d["z"], 3)
            else:
                coords = self.get_real_coords(det.center_x, det.center_y)
                if coords:
                    item["x"] = round(coords["x"], 3)
                    item["y"] = round(coords["y"], 3)
                    item["z"] = round(coords["z"], 3)
                    item["depth"] = round(coords["depth"], 3)
            results.append(item)
        self._scan_cache = results
        self._scan_cache_time = now
        return results

    def _filter_by_prompts(self, dets: list, prompts: str) -> list:
        import re
        patterns = re.split(r"[,\.\;]+", prompts)
        patterns = [p.strip().lower() for p in patterns if p.strip()]
        if not patterns:
            return dets
        matched = []
        for det in dets:
            for p in patterns:
                if p in det.name.lower():
                    matched.append(det)
                    break
        return matched

    def _process_raw_frame(self, raw: np.ndarray) -> np.ndarray:
        raw = cv2.rotate(raw, cv2.ROTATE_180)
        half_w = raw.shape[1] // 2

        if self.depth_enabled or self.hud_enabled or self.detection_enabled:
            imgL = cv2.remap(raw[:, :half_w], self.lMapX, self.lMapY, cv2.INTER_LINEAR)
            imgR = cv2.remap(raw[:, half_w:], self.rMapX, self.rMapY, cv2.INTER_LINEAR)
            frame = imgL if self.show_left else imgR
            frame = cv2.resize(frame, self.img_size)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            grayL_d = cv2.resize(grayL, self.img_size, interpolation=cv2.INTER_LINEAR)
            grayR_d = cv2.resize(grayR, self.img_size, interpolation=cv2.INTER_LINEAR)

            result = self.depth_provider.compute(grayL_d, grayR_d)
            disp_final = result.disparity
            self.disp_buffer.append(disp_final.copy())
            self._latest_left = frame.copy()

            clean_frame = frame.copy() if self.detection_enabled else None

            if self.depth_enabled:
                depth_vis = cv2.normalize(disp_final, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                depth_vis = cv2.morphologyEx(depth_vis, cv2.MORPH_OPEN, self.kernel)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.addWeighted(frame, 1 - self.alpha_depth, depth_vis, self.alpha_depth, 0, dst=frame)

        else:
            rawL = raw[:, :half_w]
            rawR = raw[:, half_w:]
            side = rawL if self.show_left else rawR
            frame = cv2.resize(side, self.img_size)

        if self.detection_enabled and self.detector.available:
            self._detection_frame_count += 1
            if self._detection_frame_count >= self._detection_frame_skip:
                self._detection_frame_count = 0
                det_frame = clean_frame if clean_frame is not None else frame
                self._last_detections = self.detector.detect(det_frame)
                valid = []
                for det in self._last_detections:
                    cx, cy = det.center_x, det.center_y
                    if 0 <= cy < frame.shape[0] and 0 <= cx < frame.shape[1]:
                        if frame[cy, cx].sum() > 0:
                            valid.append(det)
                self._last_detections = valid

            if self._last_detections:
                if self.depth_enabled and len(self.disp_buffer) > 0:
                    labels = []
                    for det in self._last_detections:
                        m3d = self._get_mask_3d(det.mask) if det.mask is not None else None
                        if m3d:
                            labels.append(f"{det.name} {det.confidence:.0%} {m3d['x']:.1f} {m3d['y']:.1f} {m3d['z']:.1f} [{m3d['vx']:.2f}x{m3d['vy']:.2f}x{m3d['vz']:.2f}]")
                        else:
                            coords = self.get_real_coords(det.center_x, det.center_y)
                            if coords:
                                labels.append(f"{det.name} {det.confidence:.0%} {coords['x']:.1f} {coords['y']:.1f} {coords['z']:.1f}")
                            else:
                                labels.append(f"{det.name} {det.confidence:.0%}")
                    frame = self.detector.annotate(frame, self._last_detections, labels=labels)
                else:
                    frame = self.detector.annotate(frame, self._last_detections)

        return frame

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
