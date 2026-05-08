import cv2
import numpy as np
import json
import threading
import time
import platform
from datetime import datetime

def log(message):
    """Log message with timestamp for Logs tab."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CAM] {message}")

class StereoCamera:
    def __init__(self, config_path, source=0):
        log(f"=== StereoCamera Initialization Started ===")
        log(f"Config path: {config_path}")
        log(f"Camera source: {source}")
        
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.source = source
        self.config_path = config_path
        
        log(f"Loading image size from config...")
        self.img_size = tuple(cfg['imSize'])
        log(f"Image size set to: {self.img_size}")
        self.depth_scale = 0.35
        self.low_size = (int(self.img_size[0] * self.depth_scale),
                         int(self.img_size[1] * self.depth_scale))
        self.Kl = np.array(cfg['Kl'])
        self.Dl = np.array(cfg['Dl'])
        self.Kr = np.array(cfg['Kr'])
        self.Dr = np.array(cfg['Dr'])
        self.R = np.array(cfg['R'])
        self.T = np.array(cfg['T'])

        self.R1, self.R2, self.P1, self.P2, self.Q = cv2.fisheye.stereoRectify(
            self.Kl, self.Dl, self.Kr, self.Dr, self.img_size, self.R, self.T, flags=0)
        self.mapL1, self.mapL2 = cv2.fisheye.initUndistortRectifyMap(
            self.Kl, self.Dl, self.R1, self.P1, self.img_size, cv2.CV_16SC2)
        self.mapR1, self.mapR2 = cv2.fisheye.initUndistortRectifyMap(
            self.Kr, self.Dr, self.R2, self.P2, self.img_size, cv2.CV_16SC2)

        log(f"Computing stereo rectification...")
        self.Q_low = self.Q.copy()
        self.Q_low[:2, :3] *= self.depth_scale
        log(f"Stereo rectification complete")

        log(f"Setting up camera parameters...")
        self.num_disp = 8
        self.block_size = 7
        self.alpha_depth = 0.3
        self.show_left = True
        self.depth_enabled = False  # Disabled by default for performance
        log(f"Depth scale: {self.depth_scale}, Low size: {self.low_size}")

        self.wls_enabled = True
        self.wls_lambda = 10000
        self.wls_sigma = 1.5
        log(f"WLS filter enabled: {self.wls_enabled}, lambda: {self.wls_lambda}, sigma: {self.wls_sigma}")

        self.ema_alpha = 0.3
        self.prev_disp = None
        self.prev_disp_lock = threading.Lock()
        log(f"EMA alpha: {self.ema_alpha}")

        log(f"Initializing matchers...")
        self._init_matchers()

        log(f"Setting up thread locks and buffers...")
        self.lock = threading.Lock()
        self.rectL = None
        self.rectR = None
        self.frame = None
        self.points_3d = None
        self.points_color = None
        self.fps = 0.0
        log(f"Buffers initialized")

        log(f"Creating video capture...")
        self.cap = self._create_capture()
        if not self.cap.isOpened():
            log(f"ERROR: Cannot open camera source {self.source}")
            raise IOError(f"Cannot open camera source {self.source}")
        log(f"Video capture created successfully")
        self._configure_capture()

        self.running = True
        log(f"Starting capture and processing threads...")
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._processing_loop, daemon=True).start()
        log(f"=== StereoCamera Initialization Complete ===")

    def _init_matchers(self):
        max_d = self.num_disp * 16
        log(f"Creating StereoSGBM matcher with numDisparities={max_d}, blockSize={self.block_size}")
        self.matcher_l = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max_d,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=200,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.wls_available = False
        self.matcher_r = None
        self.wls_filter = None
        try:
            log("Creating right matcher and WLS filter...")
            self.matcher_r = cv2.ximgproc.createRightMatcher(self.matcher_l)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher_l)
            self.wls_filter.setLambda(self.wls_lambda)
            self.wls_filter.setSigmaColor(self.wls_sigma)
            self.wls_available = True
            log("WLS filter initialized successfully")
        except Exception as e:
            log(f"WLS not available: {e}")

    def _capture_loop(self):
        log("Capture loop started")
        reconnect_delay = 1.0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                log("Camera not available, attempting reconnection...")
                time.sleep(reconnect_delay)
                try:
                    self.cap = self._create_capture()
                    if self.cap.isOpened():
                        self._configure_capture()
                        log("Camera reconnected successfully")
                except Exception as e:
                    log(f"Reconnection failed: {e}")
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                log("Frame read failed, attempting reconnect...")
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
                continue
            
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            if frame.shape[1] == 2560 and frame.shape[0] == 720:
                imgL = frame[:, :1280]
                imgR = frame[:, 1280:]
            else:
                mid = frame.shape[1] // 2
                imgL = frame[:, :mid]
                imgR = frame[:, mid:]
                imgL = cv2.resize(imgL, self.img_size)
                imgR = cv2.resize(imgR, self.img_size)
            with self.lock:
                self.rawL = imgL
                self.rawR = imgR

    def _processing_loop(self):
        log("Processing loop started")
        last_time = time.time()
        frame_count = 0
        while self.running:
            with self.lock:
                if not hasattr(self, 'rawL') or self.rawL is None:
                    time.sleep(0.01)
                    continue
                imgL = self.rawL.copy()
                imgR = self.rawR.copy()
                self.rawL = None
                self.rawR = None
            frame_count += 1
            if frame_count % 100 == 0:
                log(f"Processed {frame_count} frames, FPS: {self.fps:.1f}")

            rectL = cv2.remap(imgL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
            rectR = cv2.remap(imgR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)
            with self.lock:
                self.rectL = rectL
                self.rectR = rectR

            main_view = rectL if self.show_left else rectR

            if self.depth_enabled:
                lowL = cv2.resize(rectL, self.low_size, interpolation=cv2.INTER_AREA)
                lowR = cv2.resize(rectR, self.low_size, interpolation=cv2.INTER_AREA)
                grayL = cv2.cvtColor(lowL, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(lowR, cv2.COLOR_BGR2GRAY)

                dispL = self.matcher_l.compute(grayL, grayR).astype(np.float32) / 16.0

                if self.wls_enabled and self.wls_available:
                    dispR = self.matcher_r.compute(grayR, grayL).astype(np.float32) / 16.0
                    filtered_disp = self.wls_filter.filter(dispL, lowL, disparity_map_right=dispR)
                    filtered_disp[filtered_disp <= 0] = 0
                    current_disp = filtered_disp
                else:
                    current_disp = dispL

                with self.prev_disp_lock:
                    if self.prev_disp is not None and self.prev_disp.shape == current_disp.shape:
                        valid = (self.prev_disp > 0) & (current_disp > 0)
                        smoothed = np.where(valid,
                                           self.ema_alpha * current_disp + (1 - self.ema_alpha) * self.prev_disp,
                                           current_disp)
                    else:
                        smoothed = current_disp
                    self.prev_disp = smoothed.copy()

                points = cv2.reprojectImageTo3D(smoothed, self.Q_low)
                low_main = cv2.resize(main_view, self.low_size, interpolation=cv2.INTER_AREA)

                with np.errstate(invalid='ignore'):
                    disp_vis = np.clip(smoothed / (self.num_disp * 16) * 255, 0, 255).astype(np.uint8)
                disp_color = cv2.resize(cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA), self.img_size)
                output = cv2.addWeighted(main_view, 1.0 - self.alpha_depth, disp_color, self.alpha_depth, 0)

                with self.lock:
                    self.points_3d = points
                    self.points_color = low_main
            else:
                output = main_view
                with self.lock:
                    self.points_3d = None
                    self.points_color = None

            with self.lock:
                self.frame = output
                self.fps = 1.0 / (time.time() - last_time)
                last_time = time.time()

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                log("Frame retrieved")
            return self.frame.copy() if self.frame is not None else None

    def get_rectified_frame(self, left=True):
        side = "left" if left else "right"
        with self.lock:
            if left and self.rectL is not None:
                log(f"Rectified {side} frame retrieved")
                return self.rectL.copy()
            elif not left and self.rectR is not None:
                log(f"Rectified {side} frame retrieved")
                return self.rectR.copy()
            return None

    def get_depth_at(self, x, y):
        with self.lock:
            if self.points_3d is None:
                log(f"Depth query at ({x}, {y}) - no 3D points available")
                return None
            scale_x = self.low_size[0] / self.img_size[0]
            scale_y = self.low_size[1] / self.img_size[1]
            lx = int(x * scale_x)
            ly = int(y * scale_y)
            if lx < 0 or lx >= self.low_size[0] or ly < 0 or ly >= self.low_size[1]:
                log(f"Depth query at ({x}, {y}) - out of bounds")
                return None
            z = self.points_3d[ly, lx, 2]
            if 0 < z < 15000:
                log(f"Depth at ({x}, {y}) = {z/10.0:.1f} cm")
                return z / 10.0
            log(f"Depth at ({x}, {y}) - invalid depth value")
            return None

    def get_depth_image(self):
        """Return depth map as normalized 8-bit image. Computes depth on-demand if disabled."""
        log("Computing depth image...")
        with self.lock:
            rectL = self.rectL.copy() if self.rectL is not None else None
            rectR = self.rectR.copy() if self.rectR is not None else None
            if rectL is None or rectR is None:
                log("Cannot compute depth image - no rectified frames")
                return None
        
        # Compute depth map
        lowL = cv2.resize(rectL, self.low_size, interpolation=cv2.INTER_AREA)
        lowR = cv2.resize(rectR, self.low_size, interpolation=cv2.INTER_AREA)
        grayL = cv2.cvtColor(lowL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(lowR, cv2.COLOR_BGR2GRAY)

        dispL = self.matcher_l.compute(grayL, grayR).astype(np.float32) / 16.0

        if self.wls_enabled and self.wls_available:
            dispR = self.matcher_r.compute(grayR, grayL).astype(np.float32) / 16.0
            filtered_disp = self.wls_filter.filter(dispL, lowL, disparity_map_right=dispR)
            filtered_disp[filtered_disp <= 0] = 0
            disp = filtered_disp
        else:
            disp = dispL

        # Normalize to 0-255 for visualization
        with np.errstate(invalid='ignore'):
            disp_normalized = np.clip(disp / (self.num_disp * 16) * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap and resize to full resolution
        disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_MAGMA)
        return cv2.resize(disp_color, self.img_size)

    def update_params(self, alpha_depth=None, show_left=None, num_disp=None,
                      depth_enabled=None, wls_enabled=None):
        with self.lock:
            if alpha_depth is not None:
                log(f"Updating alpha_depth: {self.alpha_depth} -> {alpha_depth}")
                self.alpha_depth = max(0.0, min(1.0, alpha_depth))
            if show_left is not None:
                log(f"Updating show_left: {self.show_left} -> {show_left}")
                self.show_left = show_left
            if num_disp is not None and num_disp != self.num_disp:
                log(f"Updating num_disp: {self.num_disp} -> {num_disp}")
                self.num_disp = num_disp
                self._init_matchers()
                with self.prev_disp_lock:
                    self.prev_disp = None
            if depth_enabled is not None:
                log(f"Updating depth_enabled: {self.depth_enabled} -> {depth_enabled}")
                self.depth_enabled = depth_enabled
            if wls_enabled is not None:
                log(f"Updating wls_enabled: {self.wls_enabled} -> {wls_enabled}")
                self.wls_enabled = wls_enabled

    def _create_capture(self):
        """Create VideoCapture with appropriate backend for the platform."""
        if platform.system() == "Linux":
            # On Linux/Debian, explicitly use V4L2 backend
            log(f"Opening camera {self.source} with V4L2 backend...")
            
            # Check if device exists
            import os
            device_path = f"/dev/video{self.source}" if isinstance(self.source, int) else self.source
            if os.path.exists(device_path):
                log(f"Device {device_path} found")
            else:
                log(f"Warning: Device {device_path} not found")
                try:
                    import subprocess
                    result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True, timeout=5)
                    log(f"Available devices:\n{result.stdout}")
                except:
                    log("Install v4l-utils for device listing: sudo apt install v4l-utils")
            
            cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            if cap.isOpened():
                log("V4L2 backend opened camera successfully")
                return cap
            log("V4L2 failed, trying default backend...")
        
        # Fallback to default backend
        log("Trying default OpenCV backend...")
        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            log("Default backend opened camera successfully")
        else:
            log("ERROR: Default backend could not open camera")
        return cap

    def _configure_capture(self):
        """Configure capture properties."""
        log("Configuring capture properties...")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        log(f"Capture configured: 2560x720, MJPG, 30 FPS")
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        log(f"Actual capture settings: {actual_width}x{actual_height}, {actual_fps} FPS")

    def stop(self):
        log("Stopping camera...")
        self.running = False
        if self.cap:
            log("Releasing camera resource...")
            self.cap.release()
            log("Camera stopped successfully")
