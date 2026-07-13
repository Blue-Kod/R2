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
        self._max_points: int = 120000
        self._grid_cell: int = 8
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
            server_uri = rr.serve_grpc(grpc_port=self.grpc_port, server_memory_limit="500MiB")
            rr.serve_web_viewer(
                open_browser=False,
                web_port=self.port,
            )
            self._log(f"gRPC on :{self.grpc_port}, web on :{self.port}, URI: {server_uri}")

            rr.log("camera", rr.ViewCoordinates.RDF, static=True)

            self._running = True
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="rerun-log"
            )
            self._thread.start()
            return True
        except Exception as e:
            self._log(f"start failed: {e}")
            return False


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
        left, disp = self.camera.get_shared_depth()
        if left is None:
            if not self._logged_null_once:
                self._log("waiting for disp_buffer (need ≥2 frames)")
                self._logged_null_once = True
            return

        if self._logged_null_once:
            self._log("camera OK — streaming frames")
            self._logged_null_once = False

        rr.log("camera/left", rr.Image(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)))

        if self.pointcloud_enabled:
            depth, points, colors = self._build_pointcloud(disp, left)
            rr.log("camera/depth", rr.DepthImage(depth))
            if len(points) > 0:
                rr.log("camera/pointcloud", rr.Points3D(points, colors=colors))
        else:
            depth = self._disparity_to_depth(disp)
            rr.log("camera/depth", rr.DepthImage(depth))

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
        return np.clip(depth, 0, 5000).astype(np.float32)

    def _build_pointcloud(self, disp: np.ndarray, image: np.ndarray):
        h, w = disp.shape[:2]
        d16 = disp.astype(np.float32) / 16.0

        f_eff = self.camera.focal_length * h / self.camera.imSize[1]
        cx = self.camera.Kl[0, 2] * w / self.camera.imSize[0]
        cy = self.camera.Kl[1, 2] * h / self.camera.imSize[1]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright = gray > 5

        valid = (d16 > 1.0) & bright
        if not np.any(valid):
            depth = self._disparity_to_depth(disp)
            return depth, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        rows, cols = np.where(valid)
        u = cols.astype(np.float32)
        v = rows.astype(np.float32)
        d = d16[valid]

        Z = f_eff * self.camera.baseline / d * self.camera.depth_scale
        Z = np.clip(Z, 0, 5000)

        X_m = (u - cx) * Z / f_eff / 1000.0
        Y_m = (v - cy) * Z / f_eff / 1000.0
        Z_m = Z / 1000.0

        colors_bgr = image[valid]
        colors_rgb = cv2.cvtColor(
            colors_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB
        ).reshape(-1, 3)
        colors = colors_rgb.astype(np.float32) / 255.0

        points = np.column_stack([X_m, Y_m, Z_m])

        if len(points) > self._max_points:
            points, colors = self._grid_subsample(u, v, points, colors)

        depth_map = np.zeros((h, w), dtype=np.float32)
        depth_map[valid] = Z_m

        return depth_map, points.astype(np.float32), colors.astype(np.float32)

    def _grid_subsample(self, u, v, points, colors):
        cell = self._grid_cell
        u_int = u.astype(np.int32)
        v_int = v.astype(np.int32)
        cell_ids = (v_int // cell) * 10000 + u_int // cell
        _, first_idx = np.unique(cell_ids, return_index=True)
        sel = first_idx[:self._max_points]
        return points[sel], colors[sel]

    @property
    def fps(self) -> float:
        return self._fps

    def status(self) -> dict:
        return {
            "running": self._running and _HAS_RERUN,
            "available": _HAS_RERUN,
            "port": self.port,
            "grpc_port": self.grpc_port,
            "pointcloud_enabled": self.pointcloud_enabled,
            "fps": round(self._fps, 1),
        }
