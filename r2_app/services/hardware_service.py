import platform
import threading
import time
from typing import Optional

import cv2
import numpy as np

from camera import StereoCamera
from servo import ServoController


class MockStereoCamera:
    """Windows/dev fallback camera to keep web UI and APIs alive."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.img_size = (1280, 720)
        self.low_size = (320, 180)
        self.depth_enabled = True
        self.face_tracking_enabled = True
        self.tracking_mode = "person"
        self.tracking_scale_x = 50.0
        self.tracking_scale_y = 30.0
        self.tracking_offset_x = 0.0
        self.tracking_offset_y = 0.0
        self.alpha_depth = 0.3
        self.show_left = True
        self.num_disp = 5
        self.wls_enabled = False
        self.face_dx = 0.0
        self.face_dy = 0.0
        self.fps = 30.0
        self.points_3d = np.zeros((self.low_size[1], self.low_size[0], 3), dtype=np.float32)
        self._tick = 0

    def _make_frame(self) -> np.ndarray:
        frame = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        self._tick += 1
        cx = int((self._tick * 7) % self.img_size[0])
        cy = int(self.img_size[1] / 2 + np.sin(self._tick / 15.0) * 120)
        cv2.putText(frame, "SIMULATION MODE", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, "Orange Pi robot is emulated", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 40, (0, 200, 255), -1)
        return frame

    def get_frame(self):
        with self.lock:
            return self._make_frame()

    def get_depth_at(self, x, y):
        _ = (x, y)
        return 120.0

    def get_eye_offsets(self):
        with self.lock:
            return self.face_dx, self.face_dy

    def get_point_cloud_sample(self, step=2, max_distance_cm=1500):
        _ = max_distance_cm
        points = []
        for y in range(0, self.low_size[1], max(1, step)):
            for x in range(0, self.low_size[0], max(1, step)):
                points.append({"x": float(x), "y": float(y), "z": 120.0})
        return points

    def update_params(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if value is not None and hasattr(self, key):
                    setattr(self, key, value)


class MockServoController:
    def __init__(self):
        self.channel_configs = {
            0: (0, 180, 120, 520),
            1: (0, 270, 120, 520),
            2: (0, 270, 120, 520),
            3: (0, 180, 120, 520),
            4: (0, 270, 120, 520),
            5: (0, 270, 120, 520),
        }
        self.current_angles = {0: 90, 1: 135, 2: 135, 3: 90, 4: 135, 5: 135}
        self._lock = threading.Lock()

    def set_servo(self, channel, angle, smooth=True, step_delay=0.01, step_angle=2):
        _ = (smooth, step_delay, step_angle)
        if channel not in self.channel_configs:
            return False
        with self._lock:
            self.current_angles[channel] = angle
        return True


class HardwareService:
    def __init__(self, config, logger) -> None:
        self._config = config
        self._logger = logger
        self.camera: Optional[StereoCamera] = None
        self.servo: Optional[ServoController] = None
        self.servo_tracking_enabled = False
        self._tracking_thread: Optional[threading.Thread] = None

    def initialize(self) -> None:
        if self._should_run_simulation():
            self._init_simulation()
            self._start_tracking_loop()
            return
        self._init_camera()
        self._init_servo()
        if self.camera is None:
            self._logger.log("Переключаюсь в simulation mode (camera unavailable)")
            self._init_simulation()
        self._start_tracking_loop()

    def _should_run_simulation(self) -> bool:
        # Windows mode always uses simulation, Linux keeps real hardware path.
        return platform.system() == "Windows"

    def _init_simulation(self) -> None:
        self.camera = MockStereoCamera()
        self.servo = MockServoController()
        self._logger.log("Simulation mode enabled: mock camera and mock servo are active")

    def _init_camera(self) -> None:
        config_path = self._config.camera_config_path
        if not config_path.exists():
            self._logger.log(f"Файл калибровки {config_path} не найден")
            return
        try:
            self.camera = StereoCamera(str(config_path), source=self._config.camera_source)
            self._logger.log("Камера инициализирована")
        except Exception as exc:
            self._logger.log(f"Ошибка инициализации камеры: {exc}")

    def _init_servo(self) -> None:
        try:
            self.servo = ServoController(bus=0, address=0x40, freq=50)
            self._logger.log("Сервоконтроллер инициализирован")
            for channel, angle in self._config.default_servo_angles.items():
                if channel in self.servo.channel_configs:
                    self.servo.set_servo(channel, angle, smooth=False)
        except Exception as exc:
            self._logger.log(f"Ошибка инициализации сервоконтроллера: {exc}")

    def _start_tracking_loop(self) -> None:
        if self._tracking_thread is not None:
            return
        self._tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._tracking_thread.start()

    def _tracking_loop(self) -> None:
        default_neck = 90
        default_tilt = 90
        max_neck_delta = 30
        max_tilt_delta = 15
        last_neck = default_neck
        last_tilt = default_tilt
        last_target_time = time.time()
        target_lost = True

        while True:
            if self.servo_tracking_enabled and self.camera and self.servo:
                dx, dy = self.camera.get_eye_offsets()
                scale_x = self.camera.tracking_scale_x or 1.0
                scale_y = self.camera.tracking_scale_y or 1.0

                target_present = (dx != 0.0 or dy != 0.0)
                if target_present:
                    target_lost = False
                    last_target_time = time.time()

                    neck_angle = default_neck + (dx / scale_x) * max_neck_delta
                    neck_angle = max(default_neck - max_neck_delta, min(default_neck + max_neck_delta, neck_angle))
                    tilt_angle = default_tilt + (dy / scale_y) * max_tilt_delta
                    tilt_angle = max(default_tilt - max_tilt_delta, min(default_tilt + max_tilt_delta, tilt_angle))

                    if abs(neck_angle - last_neck) > 1:
                        self.servo.set_servo(0, int(round(neck_angle)), smooth=True, step_delay=0.01, step_angle=2)
                        last_neck = neck_angle
                    if abs(tilt_angle - last_tilt) > 1:
                        self.servo.set_servo(3, int(round(tilt_angle)), smooth=True, step_delay=0.01, step_angle=2)
                        last_tilt = tilt_angle
                else:
                    if not target_lost:
                        target_lost = True
                        last_target_time = time.time()
                        self._logger.log(
                            f"Цель потеряна, ожидание {self._config.tracking_timeout_seconds} секунд перед возвратом в центр"
                        )
                    elif (time.time() - last_target_time) >= self._config.tracking_timeout_seconds:
                        if abs(last_neck - default_neck) > 1:
                            self.servo.set_servo(0, default_neck, smooth=True, step_delay=0.01, step_angle=2)
                            last_neck = default_neck
                        if abs(last_tilt - default_tilt) > 1:
                            self.servo.set_servo(3, default_tilt, smooth=True, step_delay=0.01, step_angle=2)
                            last_tilt = default_tilt
            else:
                target_lost = True
                last_target_time = time.time()

            time.sleep(0.05)

