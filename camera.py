#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import threading
import time
from collections import deque

class StereoCamera:
    def __init__(self, config_path, source=0):
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.img_size = tuple(cfg['imSize'])
        self.depth_scale = 0.25
        self.low_size = (int(self.img_size[0] * self.depth_scale),
                         int(self.img_size[1] * self.depth_scale))

        self.Kl, self.Dl = np.array(cfg['Kl']), np.array(cfg['Dl'])
        self.Kr, self.Dr = np.array(cfg['Kr']), np.array(cfg['Dr'])
        self.R, self.T = np.array(cfg['R']), np.array(cfg['T'])

        self.R1, self.R2, self.P1, self.P2, self.Q = cv2.fisheye.stereoRectify(
            self.Kl, self.Dl, self.Kr, self.Dr, self.img_size, self.R, self.T, flags=0
        )

        self.mapL1, self.mapL2 = cv2.fisheye.initUndistortRectifyMap(
            self.Kl, self.Dl, self.R1, self.P1, self.img_size, cv2.CV_16SC2
        )
        self.mapR1, self.mapR2 = cv2.fisheye.initUndistortRectifyMap(
            self.Kr, self.Dr, self.R2, self.P2, self.img_size, cv2.CV_16SC2
        )

        self.Q_low = self.Q.copy()
        self.Q_low[:2, :3] *= self.depth_scale

        self.num_disp = 5
        self.block_size = 9
        self.alpha_depth = 0.3
        self.show_left = True
        self.wls_enabled = False

        self.depth_enabled = True
        self.tracking_mode = "face"   # "face" или "motion"
        self.face_tracking_enabled = True   # для обратной совместимости
        self.tracking_scale_x = 50.0
        self.tracking_scale_y = 30.0
        self.tracking_offset_x = 0.0
        self.tracking_offset_y = 0.0
        self.face_dx = 0.0
        self.face_dy = 0.0

        # MediaPipe Face Detection
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            self.use_mediapipe = True
            print("MediaPipe Face Detection инициализирован")
        except ImportError:
            print("MediaPipe не установлен, используем Haar cascade")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.use_mediapipe = False

        # Для motion detection
        self.prev_gray = None
        self.motion_center = None
        self.motion_alpha = 0.3  # сглаживание

        self._init_matchers()

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть камеру {source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.frame = None
        self.points_3d = None
        self.fps = 0
        self.running = True
        self.lock = threading.Lock()

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._processing_loop, daemon=True).start()

    def _init_matchers(self):
        max_d = self.num_disp * 16
        self.matcher_l = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=max_d, blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2, P2=32 * 3 * self.block_size ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        if self.wls_enabled:
            try:
                self.matcher_r = cv2.ximgproc.createRightMatcher(self.matcher_l)
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher_l)
                self.wls_filter.setLambda(8000)
                self.wls_filter.setSigmaColor(1.2)
                self.wls_available = True
            except AttributeError:
                print("WLS filter not available")
                self.wls_available = False
                self.matcher_r = None
        else:
            self.wls_available = False
            self.matcher_r = None

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            with self.lock:
                self.raw_frame = frame

    def _detect_face(self, image_bgr):
        """Возвращает центр лица (x, y) или None."""
        if self.use_mediapipe:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb)
            if results.detections:
                det = results.detections[0]
                bboxC = det.location_data.relative_bounding_box
                h, w, _ = image_bgr.shape
                x = int(bboxC.xmin * w + bboxC.width * w / 2)
                y = int(bboxC.ymin * h + bboxC.height * h / 2)
                return (x, y)
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
                return (x + w//2, y + h//2)
        return None

    def _detect_motion(self, gray):
        """Обнаружение движения – возвращает центр движения (x,y)."""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None
        diff = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Берём самый большой контур
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Сглаживание
                    if self.motion_center is None:
                        self.motion_center = (cx, cy)
                    else:
                        self.motion_center = (
                            int(self.motion_center[0] * (1 - self.motion_alpha) + cx * self.motion_alpha),
                            int(self.motion_center[1] * (1 - self.motion_alpha) + cy * self.motion_alpha)
                        )
                    self.prev_gray = gray.copy()
                    return self.motion_center
        self.prev_gray = gray.copy()
        return None

    def _processing_loop(self):
        last_time = time.time()
        while self.running:
            if not hasattr(self, 'raw_frame') or self.raw_frame is None:
                time.sleep(0.01)
                continue

            with self.lock:
                frame = self.raw_frame.copy()
                self.raw_frame = None

            # Разделение
            if frame.shape[1] == 2560 and frame.shape[0] == 720:
                imgL = frame[:, :1280]
                imgR = frame[:, 1280:]
            else:
                mid = frame.shape[1] // 2
                imgL = frame[:, :mid]
                imgR = frame[:, mid:]
                imgL = cv2.resize(imgL, self.img_size)
                imgR = cv2.resize(imgR, self.img_size)

            rectL = cv2.remap(imgL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
            rectR = cv2.remap(imgR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)
            main_view = rectL if self.show_left else rectR

            # ---- Трекинг цели ----
            tracking_center = None
            if self.face_tracking_enabled:   # флаг включения трекинга
                if self.tracking_mode == "face":
                    tracking_center = self._detect_face(main_view)
                elif self.tracking_mode == "motion":
                    gray = cv2.cvtColor(main_view, cv2.COLOR_BGR2GRAY)
                    tracking_center = self._detect_motion(gray)

            if tracking_center is not None:
                cx, cy = tracking_center
                # Нормализация от -1 до 1
                norm_x = (cx / self.img_size[0]) * 2 - 1
                norm_y = (cy / self.img_size[1]) * 2 - 1
                dx = -norm_x * self.tracking_scale_x + self.tracking_offset_x
                dy = norm_y * self.tracking_scale_y + self.tracking_offset_y
                dx = max(-self.tracking_scale_x * 2, min(self.tracking_scale_x * 2, dx))
                dy = max(-self.tracking_scale_y * 2, min(self.tracking_scale_y * 2, dy))
                with self.lock:
                    self.face_dx = dx
                    self.face_dy = dy
            else:
                with self.lock:
                    self.face_dx = 0.0
                    self.face_dy = 0.0

            # ---- Depth ----
            if self.depth_enabled:
                lowL = cv2.resize(rectL, self.low_size, interpolation=cv2.INTER_AREA)
                lowR = cv2.resize(rectR, self.low_size, interpolation=cv2.INTER_AREA)
                grayL = cv2.cvtColor(lowL, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(lowR, cv2.COLOR_BGR2GRAY)
                dispL = self.matcher_l.compute(grayL, grayR).astype(np.float32) / 16.0
                if self.wls_available and self.matcher_r is not None:
                    dispR = self.matcher_r.compute(grayR, grayL).astype(np.float32) / 16.0
                    filtered = self.wls_filter.filter(dispL, lowL, disparity_map_right=dispR)
                    d_float = filtered
                else:
                    d_float = dispL
                points = cv2.reprojectImageTo3D(d_float, self.Q_low)
                disp_vis = np.clip((d_float / (self.num_disp * 16)) * 255, 0, 255).astype(np.uint8)
                disp_color = cv2.resize(cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA), self.img_size)
                output = cv2.addWeighted(main_view, 1.0 - self.alpha_depth, disp_color, self.alpha_depth, 0)
                with self.lock:
                    self.points_3d = points
            else:
                output = main_view
                with self.lock:
                    self.points_3d = None

            with self.lock:
                self.frame = output
                self.fps = 1.0 / (time.time() - last_time)
                last_time = time.time()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

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

    def get_eye_offsets(self):
        with self.lock:
            return self.face_dx, self.face_dy

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
            if depth_enabled is not None:
                self.depth_enabled = depth_enabled
            if face_tracking_enabled is not None:
                self.face_tracking_enabled = face_tracking_enabled
            if tracking_mode is not None and tracking_mode in ["face", "motion"]:
                self.tracking_mode = tracking_mode
                # Сброс motion состояния при смене режима
                self.prev_gray = None
                self.motion_center = None
            if tracking_scale_x is not None:
                self.tracking_scale_x = tracking_scale_x
            if tracking_scale_y is not None:
                self.tracking_scale_y = tracking_scale_y
            if tracking_offset_x is not None:
                self.tracking_offset_x = tracking_offset_x
            if tracking_offset_y is not None:
                self.tracking_offset_y = tracking_offset_y
            if wls_enabled is not None:
                self.wls_enabled = wls_enabled
                self._init_matchers()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
