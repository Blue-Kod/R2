#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ----------------------------------------------------------------------
# Пользовательская калибровка — РЕДАКТИРУЙТЕ ЗДЕСЬ
# Offset (в градусах) прибавляется к команде для каждого канала.
# Положительное значение — физический угол больше команды,
# отрицательное — меньше.
# ----------------------------------------------------------------------
DEFAULT_OFFSETS: Dict[int, float] = {
    0: 0.0, 1: 0.0, 2: 0, 3: 0.0, 4: 0.0,
    5: 0.0, 6: 0.0, 7: 0, 8: 0.0, 9: 0.0
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
INVERTED_CHANNELS: Set[int] = {2, 4, 7, 8, 9}
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Спец-значение PCA9685 «выход навсегда LOW» — бит 12 регистра OFF (0x1000).
# Только оно реально снимает сигнал с канала и расслабляет серву: выход
# становится постоянным LOW, т.е. импульсов нет. Запись (ON=0, OFF=0) НЕ
# выключает канал (ON == OFF — недопустимая комбинация, генерация остаётся).
# Так же поступают библиотеки Adafruit: setPWM(pin, 0, 4096).
# ----------------------------------------------------------------------
FULL_OFF: int = 0x1000
# ----------------------------------------------------------------------

# Персистентная калибровка (offsets/инверсия), правится из браузера.
CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config.json"

# ----------------------------------------------------------------------
# ЕДИНЫЙ ИСТОЧНИК ПРАВДЫ по серво.
# Каналы, их имена, поза покоя (rest), поза включения (start), оффсеты,
# инверсия, физические и командные лимиты — определяются только здесь.
# arm_kinematics, data_collector, high_level, API и браузер берут значения
# отсюда (см. servo_config()).
# ----------------------------------------------------------------------
CHANNELS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

CHANNEL_NAMES: Dict[int, str] = {
    0: "Шея", 1: "Пр. плечо", 2: "Лев. плечо", 3: "Наклон",
    4: "Пов. прав.", 5: "Пов. лев.", 6: "Пр. локоть", 7: "Лев. локоть",
    8: "Пр. захват", 9: "Лев. захват",
}

# Поза покоя (reference для маппинга theta<->команда; не физический старт).
REST_POSE: Dict[int, int] = {
    0: 90, 1: 135, 2: 135, 3: 90, 4: 45,
    5: 45, 6: 180, 7: 180, 8: 90, 9: 90,
}
# Обратная совместимость: старое имя позы покоя.
DEFAULT_POSE = REST_POSE

# Поза включения (по умолчанию): применяется при старте робота.
# Руки вниз, как у человека: 90 135 135 90 45 45 180 180 90 90
START_POSE: Dict[int, int] = {
    0: 90, 1: 135, 2: 135, 3: 90, 4: 45,
    5: 45, 6: 180, 7: 180, 8: 90, 9: 90,
}
# ----------------------------------------------------------------------


def _initial_pose() -> Dict[int, int]:
    """Стартовая поза каналов — servo.START_POSE (единый источник правды)."""
    return dict(START_POSE)
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
MOVE_MAX_SPEED: float = 300.0  # крейсерская скорость, град/с
MOVE_ACCEL: float = 1000.0     # разгон/торможение, град/с^2
MOVE_DEADBAND: float = 0.5     # мёртвая зона у цели, град
# ----------------------------------------------------------------------


def servo_config() -> Dict[str, object]:
    """Полный серво-конфиг для API и фронтенда (JSON-совместимый).

    Единственная точка, откуда UI/скрипты узнают каналы, имена, лимиты,
    оффсеты, инверсию и позы. Никаких дублей этих данных в других модулях.
    """
    return {
        "order": list(CHANNELS),
        "channels": [
            {
                "id": ch,
                "name": CHANNEL_NAMES.get(ch, f"ch{ch}"),
                "min": cfg[0],
                "max": cfg[1],
                "pulse_min": cfg[2],
                "pulse_max": cfg[3],
                "command_min": DEFAULT_COMMAND_LIMITS.get(ch, cfg[:2])[0],
                "command_max": DEFAULT_COMMAND_LIMITS.get(ch, cfg[:2])[1],
                "offset": float(DEFAULT_OFFSETS.get(ch, 0.0)),
                "inverted": ch in INVERTED_CHANNELS,
            }
            for ch, cfg in sorted(DEFAULT_CHANNEL_CONFIGS.items())
        ],
        "rest": {str(ch): int(v) for ch, v in REST_POSE.items()},
        "start": {str(ch): int(v) for ch, v in START_POSE.items()},
        "inverted": sorted(INVERTED_CHANNELS),
        "move": {
            "tick": MOVE_TICK,
            "max_speed": MOVE_MAX_SPEED,
            "accel": MOVE_ACCEL,
            "deadband": MOVE_DEADBAND,
        },
    }
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

        self.current_angles: Dict[int, int] = _initial_pose()
        self.lock: threading.Lock = threading.Lock()
        # I2C/шина PCA9685 не потокобезопасна: одновременные записи из
        # mover-потоков каналов дают [Errno 22] Invalid argument. Все записи в
        # шину сериализуем одним мьютексом.
        self._bus_lock: threading.Lock = threading.Lock()
        # Восстановление связи с PCA9685: шина на Orange Pi иногда «отваливается»
        # (нет ACK), и тогда нужен повторный open + set_pwm_freq.
        self._reinit_lock: threading.Lock = threading.Lock()
        self._last_reinit: float = 0.0
        self._last_error: str = ""
        self._watchdog_started: bool = False

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

        # Persisted calibration (config.json) overrides the defaults above.
        self.load_calibration()

        self._open_pwm()
        if not self.initialized and self._chip_available():
            # Шина может подняться позже (после переподключения питания/шлейфа).
            self._start_watchdog()

    # ------------------------------------------------------------------
    # Шина PCA9685: открытие и восстановление
    # ------------------------------------------------------------------
    @staticmethod
    def _chip_available() -> bool:
        try:
            import importlib
            importlib.import_module("PCA9685_smbus2")
            return True
        except Exception:
            return False

    def _open_pwm(self) -> bool:
        try:
            from PCA9685_smbus2 import PCA9685
            self.pwm = PCA9685.PCA9685(interface=self.bus, address=self.address)
            self.pwm.set_pwm_freq(self.freq)
            self.initialized = True
            print(f"PCA9685 инициализирована на шине {self.bus}, адрес {hex(self.address)}")
            return True
        except Exception as e:
            self.pwm = None
            self.initialized = False
            self._last_error = str(e)
            print(f"Не удалось инициализировать PCA9685: {e}")
            return False

    def _close_pwm(self) -> None:
        pwm, self.pwm = self.pwm, None
        if pwm is None:
            return
        for name in ("close", "deinit"):
            fn = getattr(pwm, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

    def reinit(self) -> bool:
        """Переоткрыть шину PCA9685 (лечит «отвалившуюся» I2C)."""
        return self._reinit_pwm(min_interval=0.0)

    def _reinit_pwm(self, min_interval: float = 2.0, restore: bool = True) -> bool:
        now = time.monotonic()
        with self._reinit_lock:
            if now - self._last_reinit < min_interval:
                return self.initialized
            self._last_reinit = now
            self._close_pwm()
            ok = self._open_pwm()
            if ok and restore:
                self._restore_pose()
            elif not ok:
                print(f"[Servo] PCA9685 reinit failed: {self._last_error}")
            return ok

    def _restore_pose(self) -> None:
        """После восстановления шины вернуть сервы в известную позу.

        Если сервы расслаблены (relax_all), позу НЕ восстанавливаем — иначе
        watchdog и fallback в _set_servo_immediate снова их запитывают.
        """
        if not self._enabled:
            return
        for ch, cmd in list(self.current_angles.items()):
            if ch not in self.channel_configs:
                continue
            try:
                physical = self._physical_command(ch, int(cmd))
                self._write_pwm(ch, self.angle_to_pulse(physical, ch))
            except Exception:
                pass

    def _start_watchdog(self) -> None:
        if self._watchdog_started:
            return
        self._watchdog_started = True

        def loop() -> None:
            while True:
                time.sleep(5.0)
                if not self.initialized:
                    self._reinit_pwm(min_interval=5.0)

        threading.Thread(target=loop, daemon=True, name="servo-i2c-watchdog").start()

    def _write_pwm(self, channel: int, pulse: int) -> bool:
        if self.pwm is None:
            return False
        try:
            with self._bus_lock:
                self.pwm.set_pwm(channel, 0, pulse)
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def _write_off(self, channel: int) -> bool:
        """Снять сигнал с канала: OFF=0x1000 (full-off) → выход всегда LOW.

        Именно это расслабляет серву. Запись (ON=0, OFF=0) канал НЕ
        выключает, поэтому серва продолжает держать нагрузку.
        """
        if self.pwm is None:
            return False
        try:
            with self._bus_lock:
                self.pwm.set_pwm(channel, 0, FULL_OFF)
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def is_connected(self) -> bool:
        return bool(self.initialized and self.pwm is not None)

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
    # Персистентная калибровка (config.json)
    # ------------------------------------------------------------------
    def calibration(self) -> Dict[str, object]:
        """Текущие offsets и инвертированные каналы (для API/UI)."""
        with self.lock:
            return {
                "offsets": {str(ch): float(self.offsets.get(ch, 0.0))
                            for ch in sorted(self.channel_configs)},
                "inverted": sorted(self.inverted_channels),
            }

    def positions(self) -> Dict[str, int]:
        """Фактическое (сглаженное) положение каналов — для живого индикатора."""
        with self.lock:
            return {
                str(ch): int(round(self._move_positions.get(
                    ch, float(self.current_angles.get(ch, 0)))))
                for ch in sorted(self.channel_configs)
            }

    def load_calibration(self, path: Path = CONFIG_PATH) -> bool:
        """Применить offsets/инверсию из config.json (если файл есть)."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return False
        try:
            offsets = data.get("offsets") or {}
            for ch_str, value in offsets.items():
                ch = int(ch_str)
                if ch in self.channel_configs:
                    self.offsets[ch] = float(value)
            inverted = data.get("inverted")
            if isinstance(inverted, list):
                self.inverted_channels = {int(c) for c in inverted
                                          if int(c) in self.channel_configs}
            print(f"[Servo] Калибровка загружена из {path}")
            return True
        except Exception as e:
            print(f"[Servo] Ошибка чтения калибровки: {e}")
            return False

    def save_calibration(self, path: Path = CONFIG_PATH) -> bool:
        """Записать текущие offsets/инверсию в config.json."""
        try:
            payload = {
                "offsets": {str(ch): float(self.offsets.get(ch, 0.0))
                            for ch in sorted(self.channel_configs)},
                "inverted": sorted(self.inverted_channels),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[Servo] Калибровка сохранена в {path}")
            return True
        except Exception as e:
            print(f"[Servo] Не удалось сохранить калибровку: {e}")
            return False

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
            with self.lock:
                self.current_angles[channel] = target_command
                self._move_positions.pop(channel, None)
            with self._get_cond(channel):
                self._move_targets.pop(channel, None)
            return self._set_servo_immediate(
                channel, self._physical_command(channel, target_command))

        # Плавный режим: цель фиксируем сразу (UI/API видят команду), а
        # фактическое положение ведёт mover-поток в _move_positions.
        self._ensure_mover(channel)
        with self.lock:
            if channel not in self._move_positions:
                self._move_positions[channel] = float(
                    self.current_angles.get(channel, target_command))
            self.current_angles[channel] = target_command
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
        # Если сервы расслаблены — не создаём/оживляем mover (закр. гонку с
        # relax_all: MOVE-поток не должен снова запитать канал).
        if not self._enabled:
            return
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

        def physical_command(command: int) -> int:
            # Offset/inversion читаются «на лету», чтобы изменения калибровки
            # применялись со следующего движения без перезапуска mover-потока.
            offset = self.offsets.get(channel, 0)
            logical = int(max(command_min, min(
                command_max, command + int(round(offset)))))
            if channel in self.inverted_channels:
                return (command_min + command_max) - logical
            return logical

        cond = self._get_cond(channel)
        profile = self._profile(channel)
        max_speed = profile["max_speed"]
        accel = profile["accel"]
        tick = profile["tick"]
        deadband = profile["deadband"]

        while not self._mover_stop.is_set():
            if not self._enabled:
                break
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
                    channel, physical_command(target))
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
                channel, physical_command(command))
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
        if self.pwm is None and not self._reinit_pwm():
            return False
        try:
            pulse = self.angle_to_pulse(physical_angle, channel)
        except Exception as e:
            print(f"Ошибка установки сервопривода {channel}: {e}")
            return False
        for attempt in range(4):
            if self._write_pwm(channel, pulse):
                with self.lock:
                    if command_angle is not None:
                        self.current_angles[channel] = command_angle
                return True
            time.sleep(0.01 * (attempt + 1))
        # Шина «отвалилась» (нет ACK) — переоткрываем и пробуем ещё раз.
        if self._reinit_pwm(min_interval=1.0) and self._write_pwm(channel, pulse):
            with self.lock:
                if command_angle is not None:
                    self.current_angles[channel] = command_angle
            return True
        print(f"Ошибка установки сервопривода {channel}: {self._last_error}")
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
        # Дать mover-потокам завершиться, чтобы не перезаписали нули.
        time.sleep(0.05)
        # Если шина «отвалилась» — пытаемся восстановить, иначе нули не уйдут.
        if not self.is_connected():
            self._reinit_pwm(min_interval=0.0, restore=False)
        if self.pwm is None:
            print("[Servo] relax_all: PCA9685 недоступна — сигнал снять нельзя")
            return
        relaxed = 0
        for ch in self.channel_configs:
            for attempt in range(3):
                if self._write_off(ch):
                    relaxed += 1
                    break
                self._reinit_pwm(min_interval=0.0, restore=False)
                time.sleep(0.01 * (attempt + 1))
        print(f"[Servo] Сервы расслаблены ({relaxed}/{len(self.channel_configs)})")

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
