#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import Dict, List, Optional, Set, Tuple

# ----------------------------------------------------------------------
# Пользовательская калибровка — РЕДАКТИРУЙТЕ ЗДЕСЬ
# Offset (в градусах) прибавляется к команде для каждого канала.
# Положительное значение — физический угол больше команды,
# отрицательное — меньше.
# ----------------------------------------------------------------------
DEFAULT_OFFSETS: Dict[int, float] = {
    0: 0.0, 1: 0, 2: 0, 3: 0.0, 4: -7.0,
    5: -9.0, 6: 0.0, 7: 0, 8: 0.0, 9: 0.0
}

# ----------------------------------------------------------------------
# Инверсия для правых сервоприводов
# Если канал в этом множестве, угол пересчитывается как max_angle - angle.
# Для стандартной конфигурации (0..270) это даёт зеркальное отражение.
# ----------------------------------------------------------------------
INVERTED_CHANNELS: Set[int] = {1, 4, 6, 8}
# ----------------------------------------------------------------------

ChannelConfig = Tuple[int, int, int, int]


class ServoError(Exception):
    pass


class ServoController:
    def __init__(
        self,
        bus: int = 0,
        address: int = 0x40,
        freq: int = 50,
        channel_configs: Optional[Dict[int, ChannelConfig]] = None
    ) -> None:
        self.bus: int = bus
        self.address: int = address
        self.freq: int = freq
        self.pwm = None
        self.initialized: bool = False

        self.current_angles: Dict[int, int] = {
            0: 90, 1: 135, 2: 135, 3: 90, 4: 45,
            5: 45, 6: 180, 7: 180, 8: 90, 9: 90
        }
        self.lock: threading.Lock = threading.Lock()
        self._move_locks: Dict[int, threading.Lock] = {}

        self.offsets: Dict[int, float] = {}
        self.inverted_channels: Set[int] = set(INVERTED_CHANNELS)

        if channel_configs is None:
            self.channel_configs: Dict[int, ChannelConfig] = {
                0: (0, 180, 120, 520), 1: (0, 270, 102, 540),
                2: (0, 270, 102, 540), 3: (0, 180, 120, 520),
                4: (0, 270, 102, 540), 5: (0, 270, 102, 540),
                6: (0, 270, 102, 540), 7: (0, 270, 102, 540),
                8: (0, 180, 120, 520), 9: (0, 180, 120, 520),
            }
        else:
            self.channel_configs = channel_configs

        for ch in self.channel_configs:
            self.offsets[ch] = float(DEFAULT_OFFSETS.get(ch, 0.0))

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

    def reset_offsets_to_default(self) -> None:
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
    # Per-channel move lock (prevents I2C bus contention)
    # ------------------------------------------------------------------
    def _get_move_lock(self, channel: int) -> threading.Lock:
        with self.lock:
            if channel not in self._move_locks:
                self._move_locks[channel] = threading.Lock()
            return self._move_locks[channel]

    # ------------------------------------------------------------------
    # Преобразование угла в импульс
    # ------------------------------------------------------------------
    def angle_to_pulse(self, angle: float, channel: int) -> int:
        if channel not in self.channel_configs:
            raise ServoError(f"Канал {channel} не сконфигурирован")
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
    def set_servo(self, channel: int, angle: int, smooth: bool = True, step_delay: float = 0.01, step_angle: int = 2) -> bool:
        if not self.initialized or self.pwm is None:
            print(f"PCA9685 не инициализирована, канал {channel} не установлен")
            return False

        # Применяем инверсию (если канал инвертирован)
        if channel in self.inverted_channels:
            min_angle, max_angle, _, _ = self.channel_configs[channel]
            angle = max_angle - (angle - min_angle)   # зеркалируем

        move_lock = self._get_move_lock(channel)
        with move_lock:
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

    def _set_servo_immediate(self, channel: int, physical_angle: float, command_angle: Optional[int] = None) -> bool:
        try:
            pulse = self.angle_to_pulse(physical_angle, channel)
            for attempt in range(3):
                try:
                    self.pwm.set_pwm(channel, 0, pulse)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(0.01 * (attempt + 1))
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
    def test_cycle(self, channels: Optional[List[int]] = None, delay: int = 1) -> None:
        if channels is None:
            channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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

    def calibrate_channel(self, channel: int, min_pulse: Optional[int] = None, max_pulse: Optional[int] = None) -> Optional[Tuple[int, int]]:
        if channel not in self.channel_configs:
            print(f"Канал {channel} не найден")
            return None
        min_angle, max_angle, old_min, old_max = self.channel_configs[channel]
        if min_pulse is not None:
            old_min = int(min_pulse)
        if max_pulse is not None:
            old_max = int(max_pulse)
        self.channel_configs[channel] = (min_angle, max_angle, old_min, old_max)
        return old_min, old_max