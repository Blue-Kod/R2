#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading

class ServoController:
    def __init__(self, bus=0, address=0x40, freq=50, channel_configs=None):
        """
        Инициализация контроллера PCA9685.
        :param bus: номер I2C шины
        :param address: адрес устройства
        :param freq: частота ШИМ (Гц)
        :param channel_configs: словарь {channel: (min_angle, max_angle, min_pulse, max_pulse)}
        """
        self.bus = bus
        self.address = address
        self.freq = freq
        self.pwm = None
        self.initialized = False

        # Текущие углы для плавного движения
        self.current_angles = {0: 90, 1: 135, 2: 135, 3: 90, 4: 45, 5: 45, 6: 135, 7:135}
        self.lock = threading.Lock()

        # Конфигурация каналов по умолчанию (6 каналов)
        if channel_configs is None:
            self.channel_configs = {
                0: (0, 180, 120, 520),   # Шея
                1: (0, 270, 120, 520),   # Правое плечо
                2: (0, 270, 120, 520),   # Левое плечо
                3: (0, 180, 120, 520),   # Наклон головы
                4: (0, 270, 120, 520),   # Поворот правого плеча
                5: (0, 270, 120, 520),   # Поворот левого плеча
                6: (0, 270, 120, 520),   # Правый локоть
                7: (0, 270, 120, 520)    # Левый локоть
            }
        else:
            self.channel_configs = channel_configs

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

    def angle_to_pulse(self, angle, channel):
        if channel not in self.channel_configs:
            raise ValueError(f"Канал {channel} не сконфигурирован")
        min_angle, max_angle, min_pulse, max_pulse = self.channel_configs[channel]
        if angle < min_angle:
            angle = min_angle
        if angle > max_angle:
            angle = max_angle
        pulse = min_pulse + (max_pulse - min_pulse) * (angle - min_angle) / (max_angle - min_angle)
        return int(pulse)

    def set_servo(self, channel, angle, smooth=True, step_delay=0.01, step_angle=2):
        """
        Установить угол сервопривода на заданном канале.
        Если smooth=True, движение происходит плавно.
        Возвращает True при успехе.
        """
        if not self.initialized or self.pwm is None:
            print(f"PCA9685 не инициализирована, канал {channel} не установлен")
            return False

        with self.lock:
            current = self.current_angles.get(channel, None)
            if current is None:
                current = angle
            target = angle

        if not smooth or current == target:
            return self._set_servo_immediate(channel, target)

        # Плавное движение
        step = step_angle if target > current else -step_angle
        for cur in range(int(current), int(target), step):
            self._set_servo_immediate(channel, cur)
            time.sleep(step_delay)
        # Финальная установка
        self._set_servo_immediate(channel, target)
        return True

    def _set_servo_immediate(self, channel, angle):
        """Мгновенная установка угла (без плавности)."""
        try:
            pulse = self.angle_to_pulse(angle, channel)
            self.pwm.set_pwm(channel, 0, pulse)
            with self.lock:
                self.current_angles[channel] = angle
            # print(f"Сервопривод {channel} установлен в угол {angle}° (импульс {pulse})")
            return True
        except Exception as e:
            print(f"Ошибка установки сервопривода {channel}: {e}")
            return False

    def test_cycle(self, channels=None, delay=1):
        if channels is None:
            channels = [0, 1, 2, 3, 4, 5]
        if not self.initialized:
            return
        for ch in channels:
            if ch not in self.channel_configs:
                continue
            min_angle, max_angle, _, _ = self.channel_configs[ch]
            mid = (min_angle + max_angle) // 2
            angles = [min_angle, mid, max_angle]
            for angle in angles:
                self.set_servo(ch, angle, smooth=True, step_delay=0.02, step_angle=3)
                time.sleep(delay)
            time.sleep(1)

    def calibrate_channel(self, channel, min_pulse=None, max_pulse=None):
        if channel not in self.channel_configs:
            print(f"Канал {channel} не найден")
            return
        min_angle, max_angle, old_min, old_max = self.channel_configs[channel]
        if min_pulse is not None:
            old_min = int(min_pulse)
        if max_pulse is not None:
            old_max = int(max_pulse)
        self.channel_configs[channel] = (min_angle, max_angle, old_min, old_max)
        return old_min, old_max
