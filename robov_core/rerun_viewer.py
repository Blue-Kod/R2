import threading
import time
from typing import Optional

import cv2
import numpy as np

try:
    import rerun as rr
    _HAS_RERUN = True
except ImportError:
    _HAS_RERUN = False

from robov_core.high_level import ip_address

_LOG_TAG = "[RerunViewer]"


class RerunViewer:
    WEB_VIEWER_PORT = 9876
    GRPC_PORT = 9877

    def __init__(self, camera, port: int = WEB_VIEWER_PORT, log_fn=None):
        self.camera = camera
        self.port = port
        self.grpc_port: int = self.GRPC_PORT
        self.pointcloud_enabled: bool = True
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._downscale: int = 2
        self._max_points: int = 50000
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._last_fps_time: float = time.time()
        self._log_fn = log_fn or print
        self._logged_null_once: bool = False

    def _log(self, msg: str) -> None:
        self._log_fn(f"{_LOG_TAG} {msg}")

    def start(self) -> bool:
        if not _HAS_RERUN:
            self._log("rerun-sdk not installed")
            return False
        try:
            rr.init("r2_robot")
            server_uri = rr.serve_grpc(grpc_port=self.grpc_port)
            # Replace 127.0.0.1 with real IP so remote browsers connect to robot
            host_ip = ip_address()
            remote_uri = server_uri.replace("127.0.0.1", host_ip)
            rr.serve_web_viewer(
                open_browser=False,
                web_port=self.port,
                connect_to=remote_uri,
            )
            self._log(f"gRPC on :{self.grpc_port}, web on :{self.port}, URI: {remote_uri}")

            self._send_test_frame()

            self._running = True
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="rerun-log"
            )
            self._thread.start()
            return True
        except Exception as e:
            self._log(f"start failed: {e}")
            return False

    def _send_test_frame(self) -> None:
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.putText(img, "R2 Connected", (10, 70),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        rr.log("camera/status", rr.Image(img))

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _loop(self) -> None:
        while self._running:
            try:
                self._log_frame()
            except Exception as e:
                self._log(f"_log_frame error: {e}")
            time.sleep(0.05)

    def _log_frame(self) -> None:
        data = self.camera.capture_frame_with_depth()
        if data is None:
            if not self._logged_null_once:
                self._log("camera returned None — no frame logged yet")
                self._logged_null_once = True
            return

        if self._logged_null_once:
            self._log("camera OK — streaming frames")
            self._logged_null_once = False

        left = data["left_frame"]
        disp = data["disparity_map"]

        h, w = left.shape[:2]
        nw, nh = w // self._downscale, h // self._downscale
        left_small = cv2.resize(left, (nw, nh))
        disp_small = cv2.resize(disp, (nw, nh))

        rr.log("camera/left", rr.Image(cv2.cvtColor(left_small, cv2.COLOR_BGR2RGB)))

        depth = self._disparity_to_depth(disp_small)
        rr.log("camera/depth", rr.DepthImage(depth.astype(np.float32)))

        if self.pointcloud_enabled:
            points, colors = self._build_pointcloud(disp_small, left_small)
            if len(points) > 0:
                rr.log("camera/pointcloud", rr.Points3D(points, colors=colors))

        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_time >= 1.0:
            self._fps = self._frame_count / (now - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = now
            rr.log("system/rerun_fps", rr.Scalars(self._fps))

    def _disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        d16 = disp.astype(np.float32) / 16.0
        f_eff = self.camera.focal_length * disp.shape[0] / self.camera.imSize[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(
                d16 > 1.0,
                f_eff * self.camera.baseline / d16,
                0.0,
            ) * self.camera.depth_scale
        return np.clip(depth, 0, 50000)

    def _build_pointcloud(self, disp: np.ndarray, image: np.ndarray):
        disp_float = disp.astype(np.float32) / 16.0
        points_3d = cv2.reprojectImageTo3D(disp_float, self.camera.Q)

        valid = disp_float > 1.0
        points = points_3d[valid] * self.camera.depth_scale / 1000.0

        colors_bgr = image[valid]
        colors_rgb = cv2.cvtColor(
            colors_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB
        ).reshape(-1, 3)
        colors = colors_rgb.astype(np.float32) / 255.0

        if len(points) > self._max_points:
            idx = np.random.choice(len(points), self._max_points, replace=False)
            points = points[idx]
            colors = colors[idx]

        return points.astype(np.float32), colors.astype(np.float32)

    @property
    def fps(self) -> float:
        return self._fps

    def status(self) -> dict:
        return {
            "running": self._running and _HAS_RERUN,
            "available": _HAS_RERUN,
            "port": self.port,
            "pointcloud_enabled": self.pointcloud_enabled,
            "fps": round(self._fps, 1),
        }
