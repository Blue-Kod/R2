#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import json
import threading
import time

class SimpleStereoFix:
    def __init__(self, config_path, source=0):
        """
        Захватывает кадры со стереокамеры, применяет ректификацию и сохраняет последний кадр.
        """
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.img_size = tuple(cfg['imSize'])  # (1280, 720)

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

        # Открытие камеры
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть камеру {source}")

        # Установка параметров
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Проверка реального разрешения
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Реальное разрешение камеры: {w} x {h}")

        self.frame = None          # последний обработанный кадр
        self.running = True
        self.lock = threading.Lock()

        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        """Фоновый захват и обработка кадров."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Поворот на 180°, если камера установлена вверх ногами
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Разделение на левый и правый кадры (ожидается 2560x720)
            if frame.shape[1] == 2560 and frame.shape[0] == 720:
                imgL = frame[:, :1280]
                imgR = frame[:, 1280:]
            else:
                # fallback на случай другого разрешения
                mid = frame.shape[1] // 2
                imgL = frame[:, :mid]
                imgR = frame[:, mid:]
                imgL = cv2.resize(imgL, self.img_size)
                imgR = cv2.resize(imgR, self.img_size)

            # Ректификация
            rectL = cv2.remap(imgL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
            rectR = cv2.remap(imgR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

            # Склеиваем обратно для отображения (как в test_light)
            output = np.hstack((rectL, rectR))

            with self.lock:
                self.frame = output

    def get_frame(self):
        """Возвращает последний обработанный кадр или None."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
