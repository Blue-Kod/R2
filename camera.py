import cv2
import numpy as np
import json
import threading
import time

class StereoCamera:
    MESH_STEP = 2  # фиксированный шаг для 3D-модели

    def __init__(self, config_path, source=0):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        self.source = source
        self.config_path = config_path
        self.img_size = tuple(cfg['imSize'])
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

        self.Q_low = self.Q.copy()
        self.Q_low[:2, :3] *= self.depth_scale

        self.num_disp = 8
        self.block_size = 7
        self.alpha_depth = 0.3
        self.show_left = True
        self.depth_enabled = True

        self.wls_enabled = True
        self.wls_lambda = 10000
        self.wls_sigma = 1.5

        self.ema_alpha = 0.3
        self.prev_disp = None
        self.prev_disp_lock = threading.Lock()

        self._init_matchers()

        self.lock = threading.Lock()
        self.rectL = None
        self.rectR = None
        self.frame = None
        self.points_3d = None
        self.points_color = None
        self.fps = 0.0
        self.face_dx = 0.0
        self.face_dy = 0.0

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera source {self.source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._processing_loop, daemon=True).start()

    def _init_matchers(self):
        max_d = self.num_disp * 16
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
            self.matcher_r = cv2.ximgproc.createRightMatcher(self.matcher_l)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher_l)
            self.wls_filter.setLambda(self.wls_lambda)
            self.wls_filter.setSigmaColor(self.wls_sigma)
            self.wls_available = True
            print("[CAM] WLS filter initialized")
        except Exception as e:
            print(f"[CAM] WLS not available: {e}")

    def _capture_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
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
        last_time = time.time()
        while self.running:
            with self.lock:
                if not hasattr(self, 'rawL') or self.rawL is None:
                    time.sleep(0.01)
                    continue
                imgL = self.rawL.copy()
                imgR = self.rawR.copy()
                self.rawL = None
                self.rawR = None

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
            return self.frame.copy() if self.frame is not None else None

    def get_rectified_frame(self, left=True):
        with self.lock:
            if left and self.rectL is not None:
                return self.rectL.copy()
            elif not left and self.rectR is not None:
                return self.rectR.copy()
            return None

    def get_eye_offsets(self):
        return 0.0, 0.0

    def get_depth_at(self, x, y):
        with self.lock:
            if self.points_3d is None:
                return None
            scale_x = self.low_size[0] / self.img_size[0]
            scale_y = self.low_size[1] / self.img_size[1]
            lx = int(x * scale_x)
            ly = int(y * scale_y)
            if lx < 0 or lx >= self.low_size[0] or ly < 0 or ly >= self.low_size[1]:
                return None
            z = self.points_3d[ly, lx, 2]
            if 0 < z < 15000:
                return z / 10.0
            return None

    def get_point_cloud_sample(self, step=2, max_distance_cm=1500):
        with self.lock:
            if self.points_3d is None:
                return []
            pts = self.points_3d
            colors = self.points_color
            h, w = pts.shape[:2]
            points = []
            for y in range(0, h, step):
                for x in range(0, w, step):
                    X, Y, Z = pts[y, x]
                    if Z <= 0 or Z > max_distance_cm * 10:
                        continue
                    r, g, b = 200, 200, 200
                    if colors is not None and y < colors.shape[0] and x < colors.shape[1]:
                        bgr = colors[y, x]
                        r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
                    points.append({
                        'x': float(X / 10),
                        'y': float(Y / 10),
                        'z': float(Z / 10),
                        'r': r, 'g': g, 'b': b
                    })
            return points

    def get_depth_mesh(self, max_distance_cm=1500):
        """
        Возвращает полигональную сетку с фиксированным шагом.
        """
        step = self.MESH_STEP
        with self.lock:
            if self.points_3d is None:
                return {'width': 0, 'height': 0, 'points': []}
            pts = self.points_3d
            colors = self.points_color
            h, w = pts.shape[:2]
            grid_w = w // step
            grid_h = h // step
            points = []
            for y in range(0, h, step):
                for x in range(0, w, step):
                    X, Y, Z = pts[y, x]
                    valid = (Z > 0) and (Z <= max_distance_cm * 10)
                    r, g, b = 200, 200, 200
                    if valid and colors is not None and y < colors.shape[0] and x < colors.shape[1]:
                        bgr = colors[y, x]
                        r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
                    points.append({
                        'x': float(X / 10),
                        'y': float(Y / 10),
                        'z': float(Z / 10) if valid else 0.0,
                        'r': r, 'g': g, 'b': b,
                        'valid': valid
                    })
            return {'width': grid_w, 'height': grid_h, 'points': points}

    def update_params(self, alpha_depth=None, show_left=None, num_disp=None,
                      depth_enabled=None, face_tracking_enabled=None,
                      tracking_mode=None,
                      tracking_scale_x=None, tracking_scale_y=None,
                      tracking_offset_x=None, tracking_offset_y=None,
                      wls_enabled=None):
        with self.lock:
            if alpha_depth is not None:
                self.alpha_depth = max(0.0, min(1.0, alpha_depth))
            if show_left is not None:
                self.show_left = show_left
            if num_disp is not None and num_disp != self.num_disp:
                self.num_disp = num_disp
                self._init_matchers()
                with self.prev_disp_lock:
                    self.prev_disp = None
            if depth_enabled is not None:
                self.depth_enabled = depth_enabled
            if wls_enabled is not None:
                self.wls_enabled = wls_enabled

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()