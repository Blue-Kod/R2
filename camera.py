#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import threading
import time

class StereoCamera:
    def __init__(self, config_path, source=0):
        """
        Захватывает стереопару, выполняет ректификацию, вычисляет карту глубины
        и предоставляет кадры с возможностью выбора глаза и наложения глубины.
        """
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.img_size = tuple(cfg['imSize'])  # (1280, 720)
        self.low_size = (self.img_size[0] // 2, self.img_size[1] // 2)  # (640, 360)

        # Матрицы калибровки
        self.Kl, self.Dl = np.array(cfg['Kl']), np.array(cfg['Dl'])
        self.Kr, self.Dr = np.array(cfg['Kr']), np.array(cfg['Dr'])
        self.R, self.T = np.array(cfg['R']), np.array(cfg['T'])

        # Ректификация для рыбий глаз
        self.R1, self.R2, self.P1, self.P2, self.Q = cv2.fisheye.stereoRectify(
            self.Kl, self.Dl, self.Kr, self.Dr, self.img_size, self.R, self.T, flags=0
        )

        # Карты ремапинга
        self.mapL1, self.mapL2 = cv2.fisheye.initUndistortRectifyMap(
            self.Kl, self.Dl, self.R1, self.P1, self.img_size, cv2.CV_16SC2
        )
        self.mapR1, self.mapR2 = cv2.fisheye.initUndistortRectifyMap(
            self.Kr, self.Dr, self.R2, self.P2, self.img_size, cv2.CV_16SC2
        )

        # Матрица Q для преобразования диспаритета в 3D (уменьшенное разрешение)
        self.Q_low = self.Q.copy()
        self.Q_low[:2, :3] *= 0.5

        # Параметры стерео матчинга
        self.num_disp = 7          # будет умножено на 16
        self.block_size = 11
        self.alpha_depth = 0.3      # прозрачность наложения глубины (0 - только видео, 1 - только глубина)
        self.show_left = True       # True - левый глаз, False - правый

        # ----- Параметры трекинга -----
        self.depth_enabled = True            # вкл/выкл вычисление глубины
        self.face_tracking_enabled = False   # вкл/выкл отслеживание лиц
        self.tracking_scale_x = 50.0         # макс. смещение по X (пиксели) при лице у края
        self.tracking_scale_y = 30.0         # макс. смещение по Y
        self.tracking_offset_x = 0.0          # смещение по X (пиксели)
        self.tracking_offset_y = 0.0          # смещение по Y
        self.face_dx = 0.0                    # текущее вычисленное смещение для глаз (X)
        self.face_dy = 0.0                    # текущее вычисленное смещение для глаз (Y)

        # Загрузка каскада Хаара для лиц
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("Предупреждение: не удалось загрузить каскад лиц. Face tracking отключён.")
            self.face_cascade = None

        self._init_matchers()

        # Открытие камеры
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть камеру {source}")

        # Установка параметров (если поддерживаются)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Проверка реального разрешения
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Реальное разрешение камеры: {w} x {h}")

        self.frame = None          # последний обработанный кадр (для показа)
        self.points_3d = None      # трёхмерные точки (в low разрешении)
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
        try:
            self.matcher_r = cv2.ximgproc.createRightMatcher(self.matcher_l)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.matcher_l)
            self.wls_filter.setLambda(8000)
            self.wls_filter.setSigmaColor(1.2)
            self.wls_available = True
        except AttributeError:
            print("WLS filter not available. Disabling WLS.")
            self.wls_available = False
            self.matcher_r = None

    def _capture_loop(self):
        """Захват сырых кадров с камеры."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Поворот на 180°, если камера установлена вверх ногами
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Сохраняем сырой кадр для обработки
            with self.lock:
                self.raw_frame = frame

    def _processing_loop(self):
        """Обработка кадров: ректификация, стерео, формирование выходного изображения."""
        last_time = time.time()

        while self.running:
            if not hasattr(self, 'raw_frame') or self.raw_frame is None:
                time.sleep(0.01)
                continue

            with self.lock:
                frame = self.raw_frame.copy()
                self.raw_frame = None  # освобождаем для захвата

            # Разделение на левый и правый кадры
            if frame.shape[1] == 2560 and frame.shape[0] == 720:
                imgL = frame[:, :1280]
                imgR = frame[:, 1280:]
            else:
                mid = frame.shape[1] // 2
                imgL = frame[:, :mid]
                imgR = frame[:, mid:]
                imgL = cv2.resize(imgL, self.img_size)
                imgR = cv2.resize(imgR, self.img_size)

            # Ректификация
            rectL = cv2.remap(imgL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
            rectR = cv2.remap(imgR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

            # Выбор основного глаза
            main_view = rectL if self.show_left else rectR

            # ---- Face Tracking (если включено) ----
            if self.face_tracking_enabled and self.face_cascade is not None:
                gray = cv2.cvtColor(main_view, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100,100))
                if len(faces) > 0:
                    # Берём самое большое лицо (по площади)
                    (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
                    face_center_x = x + w/2
                    face_center_y = y + h/2
                    # Нормализованные координаты от -1 до 1 (0 = центр кадра)
                    norm_x = (face_center_x / self.img_size[0]) * 2 - 1
                    norm_y = (face_center_y / self.img_size[1]) * 2 - 1
                    # Вычисляем смещение для глаз (знак по X инвертирован для правильного направления)
                    dx = -norm_x * self.tracking_scale_x + self.tracking_offset_x
                    dy = norm_y * self.tracking_scale_y + self.tracking_offset_y
                    # Ограничиваем, чтобы глаза не «убегали» слишком далеко
                    dx = max(-self.tracking_scale_x * 2, min(self.tracking_scale_x * 2, dx))
                    dy = max(-self.tracking_scale_y * 2, min(self.tracking_scale_y * 2, dy))
                    with self.lock:
                        self.face_dx = dx
                        self.face_dy = dy
                else:
                    with self.lock:
                        self.face_dx = 0.0
                        self.face_dy = 0.0
            # ---------------------------------------

            # ---- Depth computation (if enabled) ----
            if self.depth_enabled:
                # Стерео матчинг на половинном разрешении
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

                # Преобразование в 3D
                points = cv2.reprojectImageTo3D(d_float, self.Q_low)

                # Визуализация глубины (цветная карта)
                disp_vis = np.clip((d_float / (self.num_disp * 16)) * 255, 0, 255).astype(np.uint8)
                disp_color = cv2.resize(cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA), self.img_size)

                # Смешивание основного вида с картой глубины
                output = cv2.addWeighted(main_view, 1.0 - self.alpha_depth, disp_color, self.alpha_depth, 0)

                with self.lock:
                    self.points_3d = points
            else:
                # Глубина отключена – просто показываем видео
                output = main_view
                with self.lock:
                    self.points_3d = None
            # ----------------------------------------

            # Обновление общих данных
            with self.lock:
                self.frame = output
                self.fps = 1.0 / (time.time() - last_time)
                last_time = time.time()

    def get_frame(self):
        """Возвращает последний обработанный кадр."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_depth_at(self, x, y):
        """Возвращает расстояние в см для пикселя (x, y) на изображении (полное разрешение)."""
        with self.lock:
            if self.points_3d is None:
                return None
            # Координаты в low resolution
            scale_x = self.low_size[0] / self.img_size[0]
            scale_y = self.low_size[1] / self.img_size[1]
            lx = int(x * scale_x)
            ly = int(y * scale_y)
            if lx < 0 or lx >= self.low_size[0] or ly < 0 or ly >= self.low_size[1]:
                return None
            z = self.points_3d[ly, lx, 2]  # Z в миллиметрах
            if 0 < z < 15000:
                return z / 10.0  # в см
            return None

    def get_eye_offsets(self):
        """Возвращает текущие смещения для глаз (dx, dy) в пикселях."""
        with self.lock:
            return self.face_dx, self.face_dy

    def update_params(self, alpha_depth=None, show_left=None, num_disp=None,
                      depth_enabled=None, face_tracking_enabled=None,
                      tracking_scale_x=None, tracking_scale_y=None,
                      tracking_offset_x=None, tracking_offset_y=None):
        """Обновление параметров."""
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
            if tracking_scale_x is not None:
                self.tracking_scale_x = tracking_scale_x
            if tracking_scale_y is not None:
                self.tracking_scale_y = tracking_scale_y
            if tracking_offset_x is not None:
                self.tracking_offset_x = tracking_offset_x
            if tracking_offset_y is not None:
                self.tracking_offset_y = tracking_offset_y

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
