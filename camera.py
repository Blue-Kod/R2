#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified stereo camera module for R2 robot.
Provides camera capture, depth estimation, point cloud generation, and target tracking.
"""

import cv2
import numpy as np
import json
import threading
import time


class StereoCamera:
    """Stereo camera handler with depth estimation and point cloud generation."""
    
    def __init__(self, config_path, source=0):
        """Initialize stereo camera with configuration file."""
        with open(config_path, "r") as f:
            cfg = json.load(f)
        
        self.img_size = tuple(cfg['imSize'])
        self.low_size = (self.img_size[0] // 4, self.img_size[1] // 4)
        
        self.Kl = np.array(cfg['Kl'], dtype=np.float64)
        self.Dl = np.array(cfg['Dl'], dtype=np.float64)
        self.Kr = np.array(cfg['Kr'], dtype=np.float64)
        self.Dr = np.array(cfg['Dr'], dtype=np.float64)
        self.R = np.array(cfg['R'], dtype=np.float64)
        self.T = np.array(cfg['T'], dtype=np.float64)
        
        # Stereo rectification
        self.R1, self.R2, self.P1, self.P2, self.Q = cv2.fisheye.stereoRectify(
            self.Kl, self.Dl, self.Kr, self.Dr, self.img_size, self.R, self.T, flags=0
        )
        
        # Undistortion maps - using numeric constant CV_16SC2 = 16
        self.mapL1, self.mapL2 = cv2.fisheye.initUndistortRectifyMap(
            self.Kl, self.Dl, self.R1, self.P1, self.img_size, 16
        )
        self.mapR1, self.mapR2 = cv2.fisheye.initUndistortRectifyMap(
            self.Kr, self.Dr, self.R2, self.P2, self.img_size, 16
        )
        
        # Q matrix for low resolution
        self.Q_low = self.Q.copy()
        self.Q_low[:2, :3] *= 0.25
        
        # Stereo matcher parameters
        self.num_disp = 5
        self.block_size = 9
        self._init_matcher()
        
        # Display parameters
        self.alpha_depth = 0.3
        self.show_left = True
        self.depth_enabled = True
        self.wls_enabled = False
        
        # Tracking parameters
        self.tracking_mode = "person"  # "person", "face", "motion"
        self.face_tracking_enabled = True
        self.tracking_scale_x = 50.0
        self.tracking_scale_y = 30.0
        self.tracking_offset_x = 0.0
        self.tracking_offset_y = 0.0
        
        # Tracking state
        self.face_dx = 0.0
        self.face_dy = 0.0
        self.smoothed_center = None
        self.prev_gray = None
        self.motion_center = None
        self.smooth_alpha = 0.3
        self.motion_alpha = 0.3
        self.detection_skip = 0
        self.detection_interval = 3
        
        # Initialize MediaPipe if available
        self._init_mediapipe()
        
        # Camera capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError("Cannot open camera {}".format(source))
        
        # Set camera properties using numeric constants
        # CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5, CAP_PROP_FOURCC=6
        # ROTATE_180=2, INTER_AREA=1, INTER_LINEAR=1
        self.cap.set(3, 2560)
        self.cap.set(4, 720)
        self.cap.set(6, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(5, 30)
        
        # State
        self.frame = None
        self.points_3d = None
        self.points_color = None
        self.fps = 0.0
        self.running = True
        self.lock = threading.Lock()
        
        # Start threads
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._processing_loop, daemon=True).start()
    
    def _init_mediapipe(self):
        """Initialize MediaPipe for pose and face detection."""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            self.use_mediapipe = True
            print("MediaPipe Pose + Face Detection initialized")
        except ImportError:
            print("MediaPipe not installed, using Haar cascade for faces")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.use_mediapipe = False
    
    def _init_matcher(self):
        """Initialize stereo matcher."""
        max_disp = self.num_disp * 16
        # STEREO_SGBM_MODE_SGBM_3WAY = 3
        self.matcher_l = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max_disp,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            mode=3
        )
        
        if self.wls_enabled:
            try:
                self.matcher_r = cv2.ximgproc.createRightMatcher(self.matcher_l)
                self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher_l)
                self.wls_filter.setLambda(8000)
                self.wls_filter.setSigmaColor(1.2)
                self.wls_available = True
            except Exception:
                self.wls_available = False
                self.matcher_r = None
        else:
            self.wls_available = False
            self.matcher_r = None
    
    def _capture_loop(self):
        """Capture frames from camera."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # cv2.ROTATE_180 = 2
            frame = cv2.rotate(frame, 2)
            with self.lock:
                self.raw_frame = frame
    
    def _detect_person(self, image_bgr):
        """Detect person and return center point (x, y)."""
        if not self.use_mediapipe:
            return None
        
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]
            h, w = image_bgr.shape[:2]
            
            cx = int((left_shoulder.x + right_shoulder.x) * w / 2)
            cy = int((left_shoulder.y + right_shoulder.y) * h / 2)
            
            return (cx, cy)
        return None
    
    def _detect_face(self, image_bgr):
        """Detect face and return center point (x, y)."""
        if self.use_mediapipe:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb)
            if results.detections:
                det = results.detections[0]
                bboxC = det.location_data.relative_bounding_box
                h, w = image_bgr.shape[:2]
                cx = int((bboxC.xmin + bboxC.width/2) * w)
                cy = int((bboxC.ymin + bboxC.height/2) * h)
                return (cx, cy)
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
                return (x + w//2, y + h//2)
        return None
    
    def _detect_motion(self, gray):
        """Detect motion and return center point."""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None
        
        diff = cv2.absdiff(self.prev_gray, gray)
        mean_diff = np.mean(diff)
        thresh_val = max(20, int(mean_diff * 1.5))
        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 800:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
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
        """Main processing loop for depth calculation and tracking."""
        last_time = time.time()
        
        while self.running:
            if not hasattr(self, 'raw_frame') or self.raw_frame is None:
                time.sleep(0.01)
                continue
            
            with self.lock:
                frame = self.raw_frame.copy()
                self.raw_frame = None
            
            # Split stereo pair
            h, w = frame.shape[:2]
            if w == 2560 and h == 720:
                imgL = frame[:, :1280]
                imgR = frame[:, 1280:]
            else:
                mid = w // 2
                imgL = frame[:, :mid]
                imgR = frame[:, mid:]
                imgL = cv2.resize(imgL, self.img_size)
                imgR = cv2.resize(imgR, self.img_size)
            
            # Rectify - INTER_LINEAR = 1
            rectL = cv2.remap(imgL, self.mapL1, self.mapL2, 1)
            rectR = cv2.remap(imgR, self.mapR1, self.mapR2, 1)
            main_view = rectL if self.show_left else rectR
            
            # Tracking
            self._process_tracking(main_view)
            
            # Depth calculation
            if self.depth_enabled:
                self._process_depth(rectL, rectR, main_view)
            else:
                with self.lock:
                    self.frame = main_view
                    self.points_3d = None
                    self.points_color = None
            
            # Update FPS
            with self.lock:
                self.fps = 1.0 / (time.time() - last_time)
                last_time = time.time()
    
    def _process_tracking(self, main_view):
        """Process target tracking."""
        if not self.face_tracking_enabled:
            return
        
        self.detection_skip += 1
        tracking_center = None
        
        if self.detection_skip >= self.detection_interval:
            self.detection_skip = 0
            
            if self.tracking_mode == "person" and self.use_mediapipe:
                tracking_center = self._detect_person(main_view)
            elif self.tracking_mode == "face":
                tracking_center = self._detect_face(main_view)
            elif self.tracking_mode == "motion":
                gray = cv2.cvtColor(main_view, cv2.COLOR_BGR2GRAY)
                tracking_center = self._detect_motion(gray)
            
            # Apply smoothing
            if tracking_center is not None:
                if self.smoothed_center is None:
                    self.smoothed_center = tracking_center
                else:
                    self.smoothed_center = (
                        int(self.smoothed_center[0] * (1 - self.smooth_alpha) + tracking_center[0] * self.smooth_alpha),
                        int(self.smoothed_center[1] * (1 - self.smooth_alpha) + tracking_center[1] * self.smooth_alpha)
                    )
        
        # Update eye offsets based on tracking
        if self.smoothed_center is not None:
            cx, cy = self.smoothed_center
            norm_x = (cx / self.img_size[0]) * 2 - 1
            norm_y = (cy / self.img_size[1]) * 2 - 1
            
            dx = -norm_x * self.tracking_scale_x + self.tracking_offset_x
            dy = norm_y * self.tracking_scale_y + self.tracking_offset_y
            
            dx = max(-self.tracking_scale_x * 2, min(self.tracking_scale_x * 2, dx))
            dy = max(-self.tracking_scale_y * 2, min(self.tracking_scale_y * 2, dy))
            
            with self.lock:
                self.face_dx = dx
                self.face_dy = dy
    
    def _process_depth(self, rectL, rectR, main_view):
        """Process depth estimation and point cloud generation."""
        # Downscale for faster processing - INTER_AREA = 1
        lowL = cv2.resize(rectL, self.low_size, interpolation=1)
        lowR = cv2.resize(rectR, self.low_size, interpolation=1)
        
        grayL = cv2.cvtColor(lowL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(lowR, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        dispL = self.matcher_l.compute(grayL, grayR).astype(np.float32) / 16.0
        
        if self.wls_available and hasattr(self, 'matcher_r') and self.matcher_r is not None:
            dispR = self.matcher_r.compute(grayR, grayL).astype(np.float32) / 16.0
            filtered = self.wls_filter.filter(dispL, lowL, disparity_map_right=dispR)
            d_float = filtered
        else:
            d_float = dispL
        
        # Generate point cloud
        points = cv2.reprojectImageTo3D(d_float, self.Q_low)
        
        # Create color visualization
        disp_vis = np.clip((d_float / (self.num_disp * 16)) * 255, 0, 255).astype(np.uint8)
        disp_color = cv2.resize(cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA), self.img_size)
        output = cv2.addWeighted(main_view, 1.0 - self.alpha_depth, disp_color, self.alpha_depth, 0)
        
        # Store point cloud with colors
        low_main = cv2.resize(main_view, self.low_size, interpolation=1)
        
        with self.lock:
            self.points_3d = points
            self.points_color = low_main
            self.frame = output
    
    def get_frame(self):
        """Get the latest processed frame."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_depth_at(self, x, y):
        """Get depth at pixel coordinates (in cm)."""
        with self.lock:
            if self.points_3d is None:
                return None
            
            # Scale coordinates to low resolution
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
        """Get current eye offset values for servo control."""
        with self.lock:
            return self.face_dx, self.face_dy
    
    def get_point_cloud_sample(self, step=2, max_distance_cm=1500):
        """Get point cloud as list of {x, y, z, r, g, b} in centimeters."""
        with self.lock:
            if self.points_3d is None or self.points_color is None:
                return []
            
            pts = self.points_3d
            colors = self.points_color
            h, w = pts.shape[:2]
            
            points = []
            max_dist_mm = max_distance_cm * 10
            
            for y in range(0, h, step):
                for x in range(0, w, step):
                    X, Y, Z = pts[y, x]
                    
                    if Z <= 0 or Z > max_dist_mm:
                        continue
                    
                    if y < colors.shape[0] and x < colors.shape[1]:
                        b, g, r = colors[y, x]
                    else:
                        r, g, b = 200, 200, 200
                    
                    points.append({
                        'x': float(X) / 10.0,
                        'y': float(Y) / 10.0,
                        'z': float(Z) / 10.0,
                        'r': int(r),
                        'g': int(g),
                        'b': int(b)
                    })
            
            return points
    
    def update_params(self, **kwargs):
        """Update camera parameters."""
        with self.lock:
            if 'alpha_depth' in kwargs and kwargs['alpha_depth'] is not None:
                self.alpha_depth = max(0.0, min(1.0, kwargs['alpha_depth']))
            
            if 'show_left' in kwargs and kwargs['show_left'] is not None:
                self.show_left = kwargs['show_left']
            
            if 'num_disp' in kwargs and kwargs['num_disp'] is not None and kwargs['num_disp'] != self.num_disp:
                self.num_disp = kwargs['num_disp']
                self._init_matcher()
            
            if 'depth_enabled' in kwargs and kwargs['depth_enabled'] is not None:
                self.depth_enabled = kwargs['depth_enabled']
            
            if 'face_tracking_enabled' in kwargs and kwargs['face_tracking_enabled'] is not None:
                self.face_tracking_enabled = kwargs['face_tracking_enabled']
                if not self.face_tracking_enabled:
                    self.smoothed_center = None
            
            if 'tracking_mode' in kwargs and kwargs['tracking_mode'] is not None:
                if kwargs['tracking_mode'] in ["person", "face", "motion"]:
                    self.tracking_mode = kwargs['tracking_mode']
                    self.smoothed_center = None
                    self.prev_gray = None
                    self.motion_center = None
            
            if 'tracking_scale_x' in kwargs and kwargs['tracking_scale_x'] is not None:
                self.tracking_scale_x = float(kwargs['tracking_scale_x'])
            
            if 'tracking_scale_y' in kwargs and kwargs['tracking_scale_y'] is not None:
                self.tracking_scale_y = float(kwargs['tracking_scale_y'])
            
            if 'tracking_offset_x' in kwargs and kwargs['tracking_offset_x'] is not None:
                self.tracking_offset_x = float(kwargs['tracking_offset_x'])
            
            if 'tracking_offset_y' in kwargs and kwargs['tracking_offset_y'] is not None:
                self.tracking_offset_y = float(kwargs['tracking_offset_y'])
            
            if 'wls_enabled' in kwargs and kwargs['wls_enabled'] is not None:
                self.wls_enabled = kwargs['wls_enabled']
                self._init_matcher()
    
    def stop(self):
        """Stop camera and release resources."""
        self.running = False
        time.sleep(0.1)
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        if hasattr(self, 'pose'):
            self.pose.close()
        
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
