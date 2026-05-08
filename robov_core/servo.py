#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading

# ----------------------------------------------------------------------
# Пользовательская калибровка — РЕДАКТИРУЙТЕ ЗДЕСЬ
# Offset (в градусах) прибавляется к команде для каждого канала.
# Положительное значение — физический угол больше команды,
# отрицательное — меньше.
# ----------------------------------------------------------------------
DEFAULT_OFFSETS = {
    0: 0.0,    # Шея
    1: -14.0,    # Правое плечо
    2: 10.0,    # Левое плечо
    3: 0.0,    # Наклон головы
    4: -7.0,    # Поворот правого плеча
    5: -9.0,    # Поворот левого плеча
    6: 0.0,    # Правый локоть
    7: 22.0,    # Левый локоть
}

# ----------------------------------------------------------------------
# Инверсия для правых сервоприводов
# Если канал в этом множестве, угол пересчитывается как max_angle - angle.
# Для стандартной конфигурации (0..270) это даёт зеркальное отражение.
# ----------------------------------------------------------------------
INVERTED_CHANNELS = {1, 4, 6}   # правое плечо, поворот правого плеча, правый локоть
# ----------------------------------------------------------------------

class ServoController:
    def __init__(self, bus=0, address=0x40, freq=50, channel_configs=None):
        self.bus = bus
        self.address = address
        self.freq = freq
        self.pwm = None
        self.initialized = False

        # Текущие углы (чистые, без offset и инверсии – логические)
        self.current_angles = {0: 90, 1: 135, 2: 135, 3: 90, 4: 45, 5: 45, 6: 135, 7: 135}
        self.lock = threading.Lock()

        self.offsets = {}
        self.inverted_channels = set(INVERTED_CHANNELS)   # копия множества

        # Конфигурация каналов по умолчанию (8 каналов)
        if channel_configs is None:
            self.channel_configs = {
                0: (0, 180, 120, 520),   # Шея (180°)
                1: (0, 270, 102, 512),   # Правое плечо (270°)
                2: (0, 270, 102, 512),   # Левое плечо (270°)
                3: (0, 180, 120, 520),   # Наклон головы (180°)
                4: (0, 270, 102, 512),   # Поворот правого плеча (270°)
                5: (0, 270, 102, 512),   # Поворот левого плеча (270°)
                6: (0, 270, 102, 512),   # Правый локоть (270°)
                7: (0, 270, 102, 512)    # Левый локоть (270°)
            }
        else:
            self.channel_configs = channel_configs

        # Заполняем offsets из DEFAULT_OFFSETS
        for ch in self.channel_configs:
            self.offsets[ch] = float(DEFAULT_OFFSETS.get(ch, 0.0))

        # Попытка подключения к PCA9685
        try:
            from PCA9685_smbus2 import PCA9685
            self.pwm = PCA9685.PCA9685(interface=self.bus, address=self.address)
            self.pwm.set_pwm_freq(self.freq)
            self.initialized = True
            print(f"PCA9685 инициализирована на шине {self.bus}, адрес {hex(self.address)}")
        except Exception as e:
            print(f"Не удалось инициализировать PCA9685: {e}")
            self.initialized = False

    # ------------------------------------------------------------------
    # Offset management
    # ------------------------------------------------------------------
    def set_offset(self, channel: int, offset: float) -> bool:
        if channel not in self.channel_configs:
            print(f"Канал {channel} не существует")
            return False
        with self.lock:
            self.offsets[channel] = offset
        return True

    def get_offset(self, channel: int) -> float:
        with self.lock:
            return self.offsets.get(channel, 0.0)

    def reset_offsets_to_default(self):
        with self.lock:
            for ch in self.channel_configs:
                self.offsets[ch] = float(DEFAULT_OFFSETS.get(ch, 0.0))
        print("Offsets reset to defaults")

    # ------------------------------------------------------------------
    # Инверсия каналов
    # ------------------------------------------------------------------
    def set_inverted(self, channel: int, inverted: bool) -> bool:
        """Включить/выключить инверсию для канала."""
        if channel not in self.channel_configs:
            return False
        with self.lock:
            if inverted:
                self.inverted_channels.add(channel)
            else:
                self.inverted_channels.discard(channel)
        return True

    def get_inverted(self, channel: int) -> bool:
        with self.lock:
            return channel in self.inverted_channels

    # ------------------------------------------------------------------
    # Преобразование угла в импульс
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Управление сервоприводом
    # ------------------------------------------------------------------
    def set_servo(self, channel, angle, smooth=True, step_delay=0.01, step_angle=2):
        if not self.initialized or self.pwm is None:
            print(f"PCA9685 не инициализирована, канал {channel} не установлен")
            return False

        # Применяем инверсию (если канал инвертирован)
        if channel in self.inverted_channels:
            min_angle, max_angle, _, _ = self.channel_configs[channel]
            angle = max_angle - (angle - min_angle)   # зеркалируем

        with self.lock:
            current = self.current_angles.get(channel, None)
            if current is None:
                current = angle
            target_command = angle
            offset = self.offsets.get(channel, 0)

        if not smooth or current == target_command:
            return self._set_servo_immediate(channel, target_command + offset, target_command)

        step = step_angle if target_command > current else -step_angle
        for cmd in range(int(current), int(target_command), step):
            self._set_servo_immediate(channel, cmd + offset, cmd)
            time.sleep(step_delay)
        self._set_servo_immediate(channel, target_command + offset, target_command)
        return True

    def _set_servo_immediate(self, channel, physical_angle, command_angle=None):
        try:
            pulse = self.angle_to_pulse(physical_angle, channel)
            self.pwm.set_pwm(channel, 0, pulse)
            with self.lock:
                if command_angle is not None:
                    self.current_angles[channel] = command_angle
            return True
        except Exception as e:
            print(f"Ошибка установки сервопривода {channel}: {e}")
            return False

    # ------------------------------------------------------------------
    # Тест и калибровка
    # ------------------------------------------------------------------
    def test_cycle(self, channels=None, delay=1):
        if channels is None:
            channels = [0, 1, 2, 3, 4, 5, 6, 7]
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