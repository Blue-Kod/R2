import numpy as np
import json
import os
import threading
import time
from collections import deque

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2


class StereoCamera:
    def __init__(self, camera_param_file="125deg_stereocam_calib_param.json", source=0):
        """
        Initialize the stereo camera with calibration parameters.
        
        Args:
            camera_param_file (str): Path to camera calibration parameters JSON file
            camera_source (int): Camera source index
        """
        self.camera_param_file = camera_param_file
        self.camera_source = source
        self.cap = None
        self.disp_buffer = deque(maxlen=5)
        self.lock = threading.Lock()
        
        # Additional properties for compatibility with high_level.py
        self.img_size = (1280, 720)
        self.low_size = (320, 180)
        self.depth_enabled = False
        self.alpha_depth = 0.3
        self.show_left = True
        self.num_disp = 5
        self.wls_enabled = False
        self.fps = 30.0
        self._tick = 0
        self._last_frame_time = time.time()
        self._frame_count = 0
        
        # StereoSGBM parameters
        self.window_size = 5
        self.min_disp = 0
        self.num_disp = 16 * 16
        
        # Load camera parameters and initialize rectification maps
        self._load_camera_parameters()
        self._setup_rectification()
        self._setup_stereo_matchers()
        
    def _load_camera_parameters(self):
        """Load camera calibration parameters from JSON file."""
        try:
            with open(self.camera_param_file) as fp:
                cp = json.load(fp)
                self.Kl, self.Dl = np.array(cp["Kl"]), np.array(cp["Dl"])
                self.Kr, self.Dr = np.array(cp["Kr"]), np.array(cp["Dr"])
                self.R, self.T = np.array(cp["R"]), np.array(cp["T"])
                self.imSize = tuple(cp["imSize"])
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            raise
            
    def _setup_rectification(self):
        """Setup stereo rectification maps."""
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
        
    def _setup_stereo_matchers(self):
        """Setup stereo matching algorithms and filters."""
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.window_size,
            P1=8 * 3 * self.window_size ** 2,
            P2=32 * 3 * self.window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=20,
            speckleWindowSize=400,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
        self.wls_filter.setLambda(8000.0)
        self.wls_filter.setSigmaColor(1.5)
        
        self.kernel = np.ones((5, 5), np.uint8)
        
    def initialize_camera(self):
        """Initialize the camera capture."""
        self.cap = cv2.VideoCapture(self.camera_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.cap.isOpened()
        
    def release_camera(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            
    def get_rectified_frames(self):
        """
        Capture and rectify stereo frames.
        
        Returns:
            tuple: (left_rectified, right_rectified) frames or (None, None) if failed
        """
        if not self.cap or not self.cap.isOpened():
            # Try to re-initialize camera
            if self.initialize_camera():
                print("Camera re-initialized successfully")
            else:
                return None, None
            
        ret, frame = self.cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not ret:
            # Try to re-initialize camera on frame read failure
            print("Frame read failed, attempting camera re-initialization...")
            if self.initialize_camera():
                print("Camera re-initialized successfully")
                # Try reading frame again
                ret, frame = self.cap.read()
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                if not ret:
                    return None, None
            else:
                return None, None
            
        half_w = frame.shape[1] // 2
        imgL = cv2.remap(frame[:, :half_w], self.lMapX, self.lMapY, cv2.INTER_LINEAR)
        imgR = cv2.remap(frame[:, half_w:], self.rMapX, self.rMapY, cv2.INTER_LINEAR)
        
        return imgL, imgR
        
    def compute_disparity(self, left_frame, right_frame):
        """
        Compute disparity map from rectified stereo frames.
        
        Args:
            left_frame: Left rectified frame
            right_frame: Right rectified frame
            
        Returns:
            numpy.ndarray: Filtered disparity map
        """
        grayL = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        displ = self.left_matcher.compute(grayL, grayR)
        dispr = self.right_matcher.compute(grayR, grayL)
        filtered_disp = self.wls_filter.filter(displ, grayL, disparity_map_right=dispr)
        filtered_disp[filtered_disp < 0] = 0
        
        # Add to buffer for averaging
        self.disp_buffer.append(filtered_disp.copy())
        
        return filtered_disp
        
    def get_depth_at_point(self, disparity_map, x=None, y=None):
        """
        Get depth measurement at a specific point or center of frame.
        
        Args:
            disparity_map: Computed disparity map
            x (int, optional): X coordinate. If None, uses center
            y (int, optional): Y coordinate. If None, uses center
            
        Returns:
            float: Depth in millimeters
        """
        points_3d = cv2.reprojectImageTo3D(disparity_map.astype(np.float32) / 16.0, self.Q)
        h, w = disparity_map.shape[:2]
        
        if x is None:
            x = w // 2
        if y is None:
            y = h // 2
            
        depth_mm = abs(points_3d[y, x][2])
        return depth_mm
        
    def get_average_depth_map(self):
        """
        Compute average depth map from buffered disparity maps.
        
        Returns:
            tuple: (average_depth_map, center_depth_mm) or (None, None) if buffer empty
        """
        if len(self.disp_buffer) == 0:
            return None, None
            
        # Average over available frames (up to 5)
        avg_disp = np.mean(self.disp_buffer, axis=0).astype(np.int16)
        
        # Get depth at center
        center_depth = self.get_depth_at_point(avg_disp)
        
        return avg_disp, center_depth
        
    def visualize_disparity(self, disparity_map):
        """
        Create visualization of disparity map.
        
        Args:
            disparity_map: Disparity map to visualize
            
        Returns:
            numpy.ndarray: Color-coded disparity visualization
        """
        disp_vis = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        disp_vis = cv2.morphologyEx(disp_vis, cv2.MORPH_OPEN, self.kernel)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        return disp_color
        
    def add_depth_text(self, frame, depth_mm, position=(30, 50)):
        """
        Add depth measurement text to frame.
        
        Args:
            frame: Frame to add text to
            depth_mm: Depth measurement in millimeters
            position (tuple): Text position (x, y)
            
        Returns:
            numpy.ndarray: Frame with depth text added
        """
        txt = f"Depth: {depth_mm:.1f} mm" if depth_mm < 5000 else "Out of range"
        result = frame.copy()
        cv2.putText(result, txt, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return result
        
    def capture_frame_with_depth(self):
        """
        Capture a single frame and compute its depth information.
        
        Returns:
            dict: Dictionary containing left_frame, right_frame, disparity_map, depth_mm
                  or None if capture failed
        """
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
        
    def clear_buffer(self):
        """Clear the disparity buffer."""
        self.disp_buffer.clear()
        
    # Additional functions for compatibility with high_level.py
    def get_rectified_frame(self, left=True):
        """
        Get a single rectified frame (left or right).
        
        Args:
            left (bool): If True, return left frame, else right frame
            
        Returns:
            numpy.ndarray: Rectified frame
        """
        left_frame, right_frame = self.get_rectified_frames()
        if left_frame is None:
            # Return black frame if camera failed
            frame = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA ERROR", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            return frame
        return left_frame if left else right_frame
        
    def get_frame(self):
        """
        Get processed frame with depth overlay if enabled.
        
        Returns:
            numpy.ndarray: Processed frame
        """
        with self.lock:
            # Update FPS calculation
            current_time = time.time()
            self._frame_count += 1
            if current_time - self._last_frame_time >= 1.0:
                self.fps = self._frame_count / (current_time - self._last_frame_time)
                self._frame_count = 0
                self._last_frame_time = current_time
            
            # Get base frame
            frame = self.get_rectified_frame(left=self.show_left)
            
            # Add depth overlay if enabled
            if self.depth_enabled:
                left_frame, right_frame = self.get_rectified_frames()
                if left_frame is not None:
                    disparity_map = self.compute_disparity(left_frame, right_frame)
                    depth_vis = self.visualize_disparity(disparity_map)
                    
                    # Resize depth visualization to match frame size
                    depth_resized = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))
                    
                    # Blend with original frame
                    frame = cv2.addWeighted(frame, 1 - self.alpha_depth, depth_resized, self.alpha_depth, 0)
                    
            return frame
            
    def get_depth_at(self, x, y):
        """
        Get depth measurement at specific coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Depth in millimeters
        """
        left_frame, right_frame = self.get_rectified_frames()
        if left_frame is None:
            return 120.0  # Default depth if camera failed
            
        disparity_map = self.compute_disparity(left_frame, right_frame)
        return self.get_depth_at_point(disparity_map, x, y)
        
            
    def get_point_cloud_sample(self, step=2, max_distance_cm=1500):
        """
        Get a sample of 3D point cloud data.
        
        Args:
            step (int): Sampling step size
            max_distance_cm (int): Maximum distance in centimeters
            
        Returns:
            list: List of 3D points
        """
        avg_disp, _ = self.get_average_depth_map()
        if avg_disp is None:
            # Return dummy points if no depth data
            points = []
            for y in range(0, self.low_size[1], max(1, step)):
                for x in range(0, self.low_size[0], max(1, step)):
                    points.append({"x": float(x), "y": float(y), "z": 120.0})
            return points
            
        # Generate 3D points from disparity map
        points_3d = cv2.reprojectImageTo3D(avg_disp.astype(np.float32) / 16.0, self.Q)
        points = []
        
        max_distance_mm = max_distance_cm * 10
        
        for y in range(0, self.low_size[1], max(1, step)):
            for x in range(0, self.low_size[0], max(1, step)):
                px, py, pz = points_3d[y, x]
                if pz > 0 and pz < max_distance_mm:
                    points.append({"x": float(px), "y": float(py), "z": float(pz)})
                    
        return points
        
    def get_real_coords(self, x_px, y_px):
        """
        Convert pixel coordinates to real-world coordinates in meters.
        Simplified implementation using existing depth methods.
        
        Args:
            x_px (int): X coordinate in pixels
            y_px (int): Y coordinate in pixels
            
        Returns:
            dict: Dictionary containing real-world coordinates {'x': float, 'y': float, 'z': float} in meters
                  or None if conversion failed
        """
        try:
            # Use existing get_depth_at method which is proven to work
            depth_mm = self.get_depth_at(x_px, y_px)
            if depth_mm is None or depth_mm <= 0:
                print(f"get_real_coords: Invalid depth: {depth_mm} mm")
                return None
            
            # Convert pixel coordinates to normalized coordinates
            h, w = self.imSize
            if x_px < 0 or x_px >= w or y_px < 0 or y_px >= h:
                print(f"get_real_coords: Coordinates out of bounds: ({x_px}, {y_px}) for image size ({w}, {h})")
                return None
            
            # Normalize pixel coordinates to [-1, 1] range
            x_norm = (x_px - w/2) / (w/2)
            y_norm = (y_px - h/2) / (h/2)
            
            # Use camera intrinsics to calculate X, Y coordinates
            # Assuming principal point is at image center
            fx = self.Kl[0, 0]  # Focal length in x
            fy = self.Kl[1, 1]  # Focal length in y
            
            # Calculate real world X, Y from depth and normalized coordinates
            x_real = (x_px - self.Kl[0, 2]) * depth_mm / fx
            y_real = (y_px - self.Kl[1, 2]) * depth_mm / fy
            z_real = depth_mm
            
            # Convert from millimeters to meters
            return {
                'x': float(x_real) / 1000.0,
                'y': float(y_real) / 1000.0,
                'z': float(z_real) / 1000.0
            }
            
        except Exception as e:
            print(f"get_real_coords: Error - {e}")
            # Fallback: return simple coordinates based on depth
            try:
                depth_mm = self.get_depth_at(x_px, y_px)
                if depth_mm and depth_mm > 0:
                    return {
                        'x': 0.0,  # Center X
                        'y': 0.0,  # Center Y  
                        'z': float(depth_mm) / 1000.0
                    }
            except:
                pass
            return None
        
    def update_params(self, **kwargs):
        """
        Update camera parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        with self.lock:
            for key, value in kwargs.items():
                if value is not None and hasattr(self, key):
                    setattr(self, key, value)
