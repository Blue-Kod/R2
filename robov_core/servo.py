#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
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
    0: 0.0, 1: 0, 2: 0, 3: 0.0, 4: -10,
    5: 0, 6: 0.0, 7: 0, 8: 0.0, 9: 0.0
}

# ----------------------------------------------------------------------
# Инверсия для сервоприводов
# Если канал в этом множестве, угол пересчитывается как max_angle - angle.
# Для стандартной конфигурации (0..270) это даёт зеркальное отражение.
#
# Оси рук (откалибровано по факту на устройстве):
#   ch4/shoulder_z правой — инверсия (pan вправо от команды),
#   ch5/shoulder_z левой — без инверсии (зеркально к правой);
#   shoulder_x (наклон): ч1 правой БЕЗ инверсии, ч2 левой С инверсией —
#   при такой расстановке обе руки наклоняются «вперёд-вверх» от команды
#   (раньше было {1, 4, 8} — обе наклонные зеркалились, физика ехала
#   «назад-вниз»). В rest-позе углы 135 = середина 0..270, поэтому смена
#   инверсии не двигает позу покоя, а только меняет направление.
# ----------------------------------------------------------------------
INVERTED_CHANNELS: Set[int] = {2, 4, 8}
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Поза по умолчанию (логические углы на старте).
# Единый источник правды: контроллер, high_level и dev-инструменты
# берут значения отсюда.
# ----------------------------------------------------------------------
DEFAULT_POSE: Dict[int, int] = {
    0: 90, 1: 135, 2: 135, 3: 90, 4: 45,
    5: 45, 6: 180, 7: 180, 8: 90, 9: 90
}
# ----------------------------------------------------------------------

ChannelConfig = Tuple[int, int, int, int]

# ----------------------------------------------------------------------
# Конфигурация каналов по умолчанию: (min_angle, max_angle, min_pulse,
# max_pulse). Единый источник правды для диапазонов углов.
# ----------------------------------------------------------------------
DEFAULT_CHANNEL_CONFIGS: Dict[int, ChannelConfig] = {
    0: (0, 180, 120, 520), 1: (0, 270, 102, 540),
    2: (0, 270, 102, 540), 3: (0, 180, 120, 520),
    4: (0, 270, 102, 540), 5: (0, 270, 102, 540),
    6: (0, 270, 102, 540), 7: (0, 270, 102, 540),
    8: (0, 180, 120, 520), 9: (0, 180, 120, 520),
}
# ----------------------------------------------------------------------

# Логические пределы команд. Они не меняют физическую PWM-калибровку:
# шея и наклон ограничены ±45° от позы по умолчанию, локти — 0..180°.
# Сервоприводы остаются откалиброванными по полным шкалам configs выше.
DEFAULT_COMMAND_LIMITS: Dict[int, Tuple[int, int]] = {
    ch: (cfg[0], cfg[1]) for ch, cfg in DEFAULT_CHANNEL_CONFIGS.items()
}
DEFAULT_COMMAND_LIMITS.update({0: (45, 135), 3: (45, 135),
                               6: (0, 180), 7: (0, 180)})
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Профиль движения (трапеция скорости): разгон, постоянная скорость,
# торможение. Единые для всех каналов; при желании переопределить —
# через ServoController.move_profile[ch] = {...}.
# ----------------------------------------------------------------------
MOVE_TICK: float = 0.01        # период управления, с (100 Гц)
MOVE_MAX_SPEED: float = 150.0  # крейсерская скорость, град/с
MOVE_ACCEL: float = 500.0      # разгон/торможение, град/с^2
MOVE_DEADBAND: float = 0.5     # мёртвая зона у цели, град
# ----------------------------------------------------------------------


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

        # Флаг «сервы запитаны». relax_all() снимает его — после этого
        # set_servo() молча отказывается, чтобы ни один фоновый поток
        # (webxr-телеоп, web-API) не мог снова запитать сервы после
        # расслабления на выключении. Возвращается только enable_all().
        self._enabled: bool = True

        self.current_angles: Dict[int, int] = dict(DEFAULT_POSE)
        self.lock: threading.Lock = threading.Lock()

        # Плавное движение: per-channel mover.
        #   _move_targets[ch]   — последняя целевая команда (None = нет задачи)
        #   _move_velocities[ch] — текущая скорость, град/с
        #   _move_positions[ch]  — фактическая (неокруглённая) позиция, град
        self._move_targets: Dict[int, Optional[int]] = {}
        self._move_velocities: Dict[int, float] = {}
        self._move_positions: Dict[int, float] = {}
        self._mover_conds: Dict[int, threading.Condition] = {}
        self._mover_threads: Dict[int, threading.Thread] = {}
        self._mover_stop = threading.Event()
        self.move_profile: Dict[int, Dict[str, float]] = {}

        self.offsets: Dict[int, float] = {}
        self.inverted_channels: Set[int] = set(INVERTED_CHANNELS)

        if channel_configs is None:
            self.channel_configs: Dict[int, ChannelConfig] = \
                dict(DEFAULT_CHANNEL_CONFIGS)
        else:
            self.channel_configs = channel_configs
        self.command_limits: Dict[int, Tuple[int, int]] = {
            ch: tuple(DEFAULT_COMMAND_LIMITS.get(ch, cfg[:2]))
            for ch, cfg in self.channel_configs.items()
        }

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
        """Move a servo using a logical command angle.

        ``current_angles`` deliberately stores the logical command from
        DEFAULT_POSE/UI, never the inverted physical PWM angle. This keeps
        servo.py, the browser and IK in one coordinate system.

        ``smooth=True`` queues the target: per-channel mover thread подводит
        серво к цели по трапеции скорости (разгон/торможение), так что
        серво не едет на максимальной скорости рывками.
        ``smooth=False`` ставит угол мгновенно.
        """
        if not self.initialized or self.pwm is None:
            print(f"PCA9685 не инициализирована, канал {channel} не установлен")
            return False
        if not self._enabled:
            return False

        command_min, command_max = self.command_limits[channel]
        target_command = int(max(command_min, min(command_max, angle)))

        if not smooth:
            with self._get_cond(channel):
                self._move_targets.pop(channel, None)
                self._move_positions.pop(channel, None)
            return self._set_servo_immediate(
                channel, self._physical_command(channel, target_command),
                target_command)

        # Плавный режим: обновляем цель и будим mover-поток (не блокируемся).
        self._ensure_mover(channel)
        with self._get_cond(channel):
            self._move_targets[channel] = target_command
            self._mover_conds[channel].notify_all()
        return True

    def _get_cond(self, channel: int) -> threading.Condition:
        with self.lock:
            if channel not in self._mover_conds:
                self._mover_conds[channel] = threading.Condition()
            return self._mover_conds[channel]

    def _ensure_mover(self, channel: int) -> None:
        with self.lock:
            thread = self._mover_threads.get(channel)
            if thread is not None and thread.is_alive():
                self._mover_stop.clear()
                return
            self._mover_stop.clear()
            mover = threading.Thread(
                target=self._mover_loop, args=(channel,),
                daemon=True, name=f"servo-mover-ch{channel}")
            self._mover_threads[channel] = mover
            mover.start()

    def _profile(self, channel: int) -> Dict[str, float]:
        default = {
            "max_speed": MOVE_MAX_SPEED,
            "accel": MOVE_ACCEL,
            "tick": MOVE_TICK,
            "deadband": MOVE_DEADBAND,
        }
        return {**default, **self.move_profile.get(channel, {})}

    def _mover_loop(self, channel: int) -> None:
        """Плавно ведёт серво к последней цели (трапеция скорости).

        Скорость растёт с ускорением ``accel`` до ``max_speed`` и тормозит
        у цели, чтобы серво мягко остановилось без рывка и перелёта.
        Новые цели просто заменяют целевую команду — никакого накопления
        потоков/целей при телеопе.
        """
        command_min, command_max = self.command_limits[channel]
        inverted = channel in self.inverted_channels
        offset = self.offsets.get(channel, 0)

        def logical_cmd(command: int) -> int:
            adjusted = command + int(round(offset))
            return int(max(command_min, min(command_max, adjusted)))

        def physical_command(command: int) -> int:
            logical = logical_cmd(command)
            if inverted:
                return (command_min + command_max) - logical
            return logical

        cond = self._get_cond(channel)
        profile = self._profile(channel)
        max_speed = profile["max_speed"]
        accel = profile["accel"]
        tick = profile["tick"]
        deadband = profile["deadband"]

        while not self._mover_stop.is_set():
            with cond:
                cond.wait_for(
                    lambda: channel in self._move_targets
                    or self._mover_stop.is_set(),
                    timeout=0.05)
                if self._mover_stop.is_set():
                    break
                target = self._move_targets.get(channel)
                if target is None:
                    continue

            with self.lock:
                current = self._move_positions.get(
                    channel, float(self.current_angles.get(channel, target)))
                velocity = self._move_velocities.get(channel, 0.0)

            distance = float(target) - current
            if abs(distance) <= deadband:
                self._set_servo_immediate(
                    channel, physical_command(target), target)
                with self.lock:
                    self._move_positions[channel] = float(target)
                    self._move_velocities[channel] = 0.0
                with cond:
                    self._move_targets.pop(channel, None)
                continue

            # Максимально допустимая скорость, чтобы успеть затормозить.
            v_brake = math.sqrt(2.0 * accel * abs(distance))
            v_target = min(max_speed, v_brake)
            v_desired = v_target if distance > 0 else -v_target

            # Разгон/торможение с ограничением accel за один тик.
            dv = v_desired - velocity
            dv = max(-accel * tick, min(accel * tick, dv))
            velocity += dv
            step = velocity * tick

            # Не проскакиваем цель.
            if abs(step) >= abs(distance):
                command = target
                velocity = 0.0
            else:
                command = int(round(current + step))
                command = int(max(command_min, min(command_max, command)))

            self._set_servo_immediate(
                channel, physical_command(command), command)
            with self.lock:
                self._move_positions[channel] = float(command)
                self._move_velocities[channel] = velocity

            time.sleep(tick)

    def _physical_command(self, channel: int, command: int) -> int:
        offset = self.offsets.get(channel, 0)
        command_min, command_max = self.command_limits[channel]
        adjusted = command + int(round(offset))
        logical = int(max(command_min, min(command_max, adjusted)))
        if channel in self.inverted_channels:
            return (command_min + command_max) - logical
        return logical

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
    # Расслабление сервоприводов
    # ------------------------------------------------------------------
    def relax_all(self) -> None:
        """Отключить PWM всех каналов — сервы перестают держать нагрузку.

        После вызова set_servo() блокируется до enable_all(): иначе любой
        фоновый поток (webxr-телеоп идёт на 8 Гц, web-API) снова запитает
        сервы через _ensure_mover()/set_pwm, и при выключении робота они
        останутся жёсткими вместо расслабления.
        """
        self._enabled = False
        self._mover_stop.set()
        for cond in list(self._mover_conds.values()):
            with cond:
                cond.notify_all()
        if not self.initialized or self.pwm is None:
            return
        for ch in self.channel_configs:
            try:
                self.pwm.set_pwm(ch, 0, 0)
            except Exception as e:
                print(f"Не удалось расслабить серво {ch}: {e}")

    def enable_all(self) -> None:
        """Разрешить управление сервами снова (снимает блок relax_all)."""
        self._enabled = True
        self._mover_stop.clear()

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
