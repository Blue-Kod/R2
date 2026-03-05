#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import threading
import time
import os

class StereoEngine:
    def __init__(self, config_path, source=0):
        print(f"Инициализация камеры, источник {source}")
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.img_size = tuple(cfg['imSize'])  # 1280x720
        self.low_size = (self.img_size[0] // 2, self.img_size[1] // 2)  # 640x360
        self.source = source
        self.config_path = config_path

        # Матрицы калибровки
        self.Kl, self.Dl = np.array(cfg['Kl']), np.array(cfg['Dl'])
        self.Kr, self.Dr = np.array(cfg['Kr']), np.array(cfg['Dr'])
        self.R, self.T = np.array(cfg['R']), np.array(cfg['T'])

        # Ректификация
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
        self.Q_low[:2, :3] *= 0.5

        self.num_disp = 7
        self.block_size = 11
        self.alpha_depth = 0.3

        self._init_matchers()

        self.raw_frame = None
        self.processed_view = None
        self.points_3d = None
        self.fps = 0
        self.running = True
        self.lock = threading.Lock()
        self.frame_count = 0
        self.error_count = 0
        self.cap = None

        self._open_camera()

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._processing_loop, daemon=True).start()

    def _open_camera(self):
        """Попытка открыть камеру с базовыми настройками."""
        if os.name == 'nt':
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_V4L2

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source, backend)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть камеру {self.source}")

        # Устанавливаем параметры захвата
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Не используем FOURCC, пусть камера сама выбирает формат
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Проверяем реальное разрешение
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Реальное разрешение камеры: {w} x {h}")

    def _init_matchers(self):
        max_d = self.num_disp * 16
        self.matcher_l = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=max_d, blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2, P2=32 * 3 * self.block_size ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        try:
            self.matcher_r = cv2.ximgproc.createRightMatcher(self.matcher_l)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher_l)
            self.wls_filter.setLambda(8000)
            self.wls_filter.setSigmaColor(1.2)
            self.wls_available = True
        except AttributeError:
            print("WLS filter not available (install opencv-contrib-python). Disabling WLS.")
            self.wls_available = False
            self.matcher_r = None

    def _capture_loop(self):
        print("Запуск захвата кадров")
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print("Камера не открыта, попытка переоткрыть...")
                try:
                    self._open_camera()
                except Exception as e:
                    print(f"Не удалось открыть камеру: {e}")
                    time.sleep(2)
                    continue

            ret, frame = self.cap.read()
            if ret:
                self.error_count = 0
                self.frame_count += 1
                # Поворот на 180°
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                self.raw_frame = frame
                if self.frame_count % 30 == 0:
                    print(f"Получен кадр #{self.frame_count}")
            else:
                self.error_count += 1
                print(f"Ошибка чтения кадра #{self.error_count}")
                if self.error_count > 10:
                    print("Слишком много ошибок, перезапуск камеры...")
                    self.cap.release()
                    self.cap = None
                    self.error_count = 0
                time.sleep(0.1)

            time.sleep(0.001)

    def _processing_loop(self):
        print("Запуск обработки")
        last_time = time.time()
        H, W = self.img_size[1], self.img_size[0]

        while self.running:
            if self.raw_frame is None:
                time.sleep(0.01)
                continue

            frame = self.raw_frame
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

            main_view = rectL

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

            with self.lock:
                self.points_3d = cv2.reprojectImageTo3D(d_float, self.Q_low)

                disp_vis = np.clip((d_float / (self.num_disp * 16)) * 255, 0, 255).astype(np.uint8)
                disp_color = cv2.resize(cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA), self.img_size)

                canvas = cv2.addWeighted(main_view, 1.0 - self.alpha_depth, disp_color, self.alpha_depth, 0)

                self.processed_view = canvas
                self.fps = 1.0 / (time.time() - last_time)
                last_time = time.time()

    def update_params(self, nD, a_depth, m_left):
        with self.lock:
            self.alpha_depth = a_depth / 100
            if nD != self.num_disp:
                self.num_disp = nD
                self._init_matchers()

    def get_depth_at(self, x, y):
        if self.points_3d is None:
            return None
        scale_x = self.low_size[0] / self.img_size[0]
        scale_y = self.low_size[1] / self.img_size[1]
        lx = int(x * scale_x)
        ly = int(y * scale_y)
        if lx < 0 or lx >= self.low_size[0] or ly < 0 or ly >= self.low_size[1]:
            return None
        z = self.points_3d[ly, lx, 2]
        if z > 0 and z < 15000:
            return z / 10.0
        return None

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
