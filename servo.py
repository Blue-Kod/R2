#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading

class ServoController:
    def __init__(self, bus=0, address=0x40, freq=50):
        """
        Инициализация контроллера PCA9685.
        :param bus: номер I2C шины (обычно 0 или 1)
        :param address: адрес устройства (по умолчанию 0x40)
        :param freq: частота ШИМ для сервоприводов (обычно 50 Гц)
        """
        self.bus = bus
        self.address = address
        self.freq = freq
        self.pwm = None
        self.initialized = False
        # Калибровочные значения импульсов для углов 0° и 270°
        # Для SG90 стандартный диапазон 0°=102, 180°=512, но можно расширить до 270°
        self.min_pulse = 102   # импульс для 0°
        self.max_pulse = 600   # импульс для 270° (подберите экспериментально)
        self._init_pca()

    def _init_pca(self):
        """Попытка подключения к PCA9685."""
        try:
            from PCA9685_smbus2 import PCA9685
            self.pwm = PCA9685.PCA9685(interface=self.bus, address=self.address)
            self.pwm.set_pwm_freq(self.freq)
            self.initialized = True
            print(f"PCA9685 инициализирована на шине {self.bus}, адрес {hex(self.address)}")
        except Exception as e:
            print(f"Не удалось инициализировать PCA9685: {e}")
            self.initialized = False

    def angle_to_pulse(self, angle, min_angle=0, max_angle=270):
        """Преобразование угла (0-270) в значение импульса (off_ticks)."""
        if angle < min_angle:
            angle = min_angle
        if angle > max_angle:
            angle = max_angle
        # Линейная интерполяция между min_pulse и max_pulse
        pulse = self.min_pulse + (self.max_pulse - self.min_pulse) * (angle - min_angle) / (max_angle - min_angle)
        return int(pulse)

    def set_servo(self, channel, angle, min_angle=0, max_angle=270):
        """
        Установить угол сервопривода на заданном канале (0-15).
        Возвращает True при успехе.
        """
        if not self.initialized or self.pwm is None:
            print(f"PCA9685 не инициализирована, канал {channel} не установлен")
            return False
        pulse = self.angle_to_pulse(angle, min_angle, max_angle)
        try:
            self.pwm.set_pwm(channel, 0, pulse)
            print(f"Сервопривод {channel} установлен в угол {angle}° (импульс {pulse})")
            return True
        except Exception as e:
            print(f"Ошибка установки сервопривода {channel}: {e}")
            return False

    def test_cycle(self, channels=[0, 1], delay=1):
        """
        Тестовый цикл: последовательно устанавливает углы 0°, 135°, 270°.
        Запускать в отдельном потоке, чтобы не блокировать сервер.
        """
        if not self.initialized:
            return
        angles = [0, 135, 270]
        for angle in angles:
            for ch in channels:
                self.set_servo(ch, angle)
                time.sleep(delay)
            time.sleep(1)

    def calibrate(self, min_pulse=None, max_pulse=None):
        """Обновление калибровочных значений импульсов."""
        if min_pulse is not None:
            self.min_pulse = int(min_pulse)
        if max_pulse is not None:
            self.max_pulse = int(max_pulse)
        return self.min_pulse, self.max_pulse
