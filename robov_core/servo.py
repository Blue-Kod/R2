#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сервы R2: четыре привода на одной оси плеча через редуктор 3:1.

Логический канал один — ``CHANNELS = (0,)``, диапазон 0..90°.
Физически каждая серва едет 1..269° (на градус не доходя до упоров),
чтобы не гудеть; логические 0..90 при этом не меняются.
Каналы PCA9685 0 и 1 — прямо, 2 и 3 — зеркально (инверсия).
Импульсы 498..2686 мкс (102..550 отсчётов при 50 Гц).

Калибровка — на каждую из четырёх серв: offset в ФИЗИЧЕСКИХ градусах и
инверсия. config.json хранит offsets 0..3 и список inverted.
"""

import json
import os
import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ----------------------------------------------------------------------
# Единственная логическая ось.
# ----------------------------------------------------------------------
CHANNELS: Tuple[int, ...] = (0,)
CHANNEL_NAMES: Dict[int, str] = {0: "Плечо"}

# Физические сервы на PCA9685 и редуктор.
SERVOS: Tuple[int, int, int, int] = (0, 1, 2, 3)
GEAR: float = 3.0
VIRTUAL_MIN, VIRTUAL_MAX = 0, 90
PHYSICAL_MIN, PHYSICAL_MAX = 0.0, 270.0
# Рабочий ход с запасом 1° от механических упоров: 1..269.
# Логика/UI по-прежнему 0..90; в упор сервы не упираются и не гудят.
PHYSICAL_SAFE_MIN, PHYSICAL_SAFE_MAX = 1.0, 269.0
# Импульсы: 102..550 отсчётов при 50 Гц = 498..2686 мкс.
# 2500 мкс (512) этим сервам не хватало: доходили только до ~250° из 270°.
# 540 (2637) дало ~268°, поэтому верх ещё чуть выше.
PULSE_US_PER_COUNT_50HZ: float = 20000.0 / 4096.0
PULSE_MIN_COUNTS_50HZ, PULSE_MAX_COUNTS_50HZ = 102, 550
PULSE_MIN_US = PULSE_MIN_COUNTS_50HZ * PULSE_US_PER_COUNT_50HZ
PULSE_MAX_US = PULSE_MAX_COUNTS_50HZ * PULSE_US_PER_COUNT_50HZ

# Поза покоя и включения: 0°. На этом экземпляре 0° стоит тихо (проверено),
# 0..90 — полный ход. Если сервы начнут гудеть в 0 — поднять до 2..3°.
REST_POSE: Dict[int, int] = {0: 0}
START_POSE: Dict[int, int] = {0: 0}

DEFAULT_OFFSETS: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
DEFAULT_INVERTED: Set[int] = {2, 3}

# (min_angle, max_angle, min_pulse, max_pulse) — в отсчётах при 50 Гц.
DEFAULT_CHANNEL_CONFIGS: Dict[int, Tuple[int, int, int, int]] = {
    0: (0, 90, 102, 550),
}
DEFAULT_COMMAND_LIMITS: Dict[int, Tuple[int, int]] = {0: (0, 90)}
# ----------------------------------------------------------------------

# Спец-значение PCA9685 «выход навсегда LOW» — снимает сигнал и расслабляет.
FULL_OFF: int = 0x1000

CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config.json"

PWM_FREQ_HZ: float = 196.8

MOVE_TICK: float = 0.01        # период управления, с
# 90 вирт.°/с = ~270 физ.°/с. Разгон/торможение — с ограничением рывка
# (jerk): без него конец хода ощущается резким ударом.
MOVE_MAX_SPEED: float = 90.0   # град/с (виртуальных)
MOVE_ACCEL: float = 200.0      # град/с² (виртуальных)
MOVE_JERK: float = 2000.0      # град/с³ (виртуальных): плавный старт/финиш
# Тормозной профиль считаем от половины доступного ускорения: дискретный
# тик 10 мс иначе не успевает погасить скорость и в конце будет рывок.
MOVE_BRAKE_FRAC: float = 0.5
MOVE_DEADBAND: float = 0.01    # мёртвая зона у цели, град


def virtual_to_pulse_us(virtual: float, offset: float = 0.0,
                        inverted: bool = False) -> float:
    """Логический угол плеча 0..90 -> импульс сервы, мкс.

    Физически едем 1..269° (на градус не доходя до упоров), поэтому
    логические 0..90 остаются как есть, но сервы не упираются и не гудят.
    offset — в физических градусах сервы, добавляется после инверсии.
    """
    v = min(VIRTUAL_MAX, max(VIRTUAL_MIN, float(virtual)))
    physical = PHYSICAL_SAFE_MIN + (
        PHYSICAL_SAFE_MAX - PHYSICAL_SAFE_MIN) * v / VIRTUAL_MAX
    if inverted:
        physical = PHYSICAL_MAX - physical
    physical = min(PHYSICAL_SAFE_MAX,
                   max(PHYSICAL_SAFE_MIN, physical + float(offset)))
    span = PULSE_MAX_US - PULSE_MIN_US
    return PULSE_MIN_US + span * physical / PHYSICAL_MAX


def pulse_us_to_counts(pulse_us: float, freq: float) -> int:
    return int(round(float(pulse_us) * float(freq) * 4096.0 / 1e6))


def servo_config() -> Dict[str, object]:
    """Полный серво-конфиг для API и фронтенда (JSON-совместимый).

    ``channels`` — логические оси управления; ``calibration_channels`` —
    физические сервы (у каждой свой offset/инверсия).
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
                "offset": 0.0,
                "inverted": False,
                "gear": GEAR,
            }
            for ch, cfg in sorted(DEFAULT_CHANNEL_CONFIGS.items())
        ],
        "calibration_channels": [
            {
                "id": servo,
                "name": f"Серва {servo + 1}",
                "offset": float(DEFAULT_OFFSETS.get(servo, 0.0)),
                "inverted": servo in DEFAULT_INVERTED,
            }
            for servo in SERVOS
        ],
        "rest": {str(ch): int(v) for ch, v in REST_POSE.items()},
        "start": {str(ch): int(v) for ch, v in START_POSE.items()},
        "inverted": sorted(DEFAULT_INVERTED),
        "move": {
            "tick": MOVE_TICK,
            "max_speed": MOVE_MAX_SPEED,
            "accel": MOVE_ACCEL,
            "jerk": MOVE_JERK,
            "deadband": MOVE_DEADBAND,
        },
    }


class ServoError(Exception):
    pass


class _SimPCA9685:
    """Заглушка PCA9685 для режима симуляции: команды принимаются в память."""

    def set_pwm(self, channel: int, on: int, off: int) -> None:
        pass

    def set_pwm_freq(self, freq: int) -> None:
        pass


class ServoController:
    def __init__(
        self,
        bus: int = 0,
        address: int = 0x40,
        freq: float = PWM_FREQ_HZ,
        simulation: bool = False,
    ) -> None:
        self.bus: int = bus
        self.address: int = address
        self.freq: float = freq
        self.simulation: bool = simulation
        self.pwm = None
        self.initialized: bool = False
        self._enabled: bool = True

        self.current_angles: Dict[int, int] = dict(START_POSE)
        self.lock: threading.Lock = threading.Lock()
        self._bus_lock: threading.Lock = threading.Lock()
        self._reinit_lock: threading.Lock = threading.Lock()
        self._last_reinit: float = 0.0
        self._last_error: str = ""
        self._watchdog_started: bool = False

        # Mover: одна логическая ось.
        self._move_targets: Dict[int, Optional[int]] = {}
        self._move_velocities: Dict[int, float] = {}
        self._move_accels: Dict[int, float] = {}
        # Длина одиночного переезда (для коротких ходов) и время последней
        # цели: поток целей от телеопа не режем, одиночные — планируем.
        self._move_spans: Dict[int, float] = {}
        self._last_target_time: Dict[int, float] = {}
        self._move_positions: Dict[int, float] = {}
        self._mover_cond = threading.Condition()
        self._mover_thread: Optional[threading.Thread] = None
        self._mover_stop = threading.Event()

        self.offsets: Dict[int, float] = dict(DEFAULT_OFFSETS)
        self.inverted: Set[int] = set(DEFAULT_INVERTED)
        self.speed_limits: Dict[int, float] = {}

        self.channel_configs: Dict[int, Tuple[int, int, int, int]] = \
            dict(DEFAULT_CHANNEL_CONFIGS)
        self.command_limits: Dict[int, Tuple[int, int]] = {
            ch: tuple(DEFAULT_COMMAND_LIMITS.get(ch, cfg[:2]))
            for ch, cfg in self.channel_configs.items()
        }

        self.load_calibration()

        self._open_pwm()
        if not self.initialized and self._chip_available():
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
        if self.simulation:
            self.pwm = _SimPCA9685()
            self.initialized = True
            print("[Servo] SIMULATION: команды применяются в память, шина I2C не трогается")
            return True
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
        """После восстановления шины вернуть плечо в известную позу."""
        if not self._enabled:
            return
        try:
            self._write_pulses(self.current_angles.get(0, 0))
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

    def _pulse_counts(self, virtual: float) -> List[Tuple[int, int]]:
        """Виртуальный угол -> [(серва, отсчёты)] для всех четырёх."""
        out: List[Tuple[int, int]] = []
        for servo in SERVOS:
            us = virtual_to_pulse_us(
                virtual,
                offset=self.offsets.get(servo, 0.0),
                inverted=servo in self.inverted,
            )
            out.append((servo, pulse_us_to_counts(us, self.freq)))
        return out

    def _write_pulses(self, virtual: float) -> bool:
        """Записать один угол всем четырём сервам под одним локом шины."""
        if self.pwm is None:
            return False
        try:
            pulses = self._pulse_counts(virtual)
            with self._bus_lock:
                for servo, counts in pulses:
                    self.pwm.set_pwm(servo, 0, counts)
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def _write_off(self, channel: int) -> bool:
        """Снять сигнал с канала: OFF=0x1000 → выход постоянно LOW."""
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
    # Калибровка: offsets и инверсия физических серв
    # ------------------------------------------------------------------
    def set_offset(self, channel: int, offset: float) -> bool:
        if channel not in SERVOS:
            print(f"Сервы {channel} не существует")
            return False
        with self.lock:
            self.offsets[channel] = float(offset)
        return True

    def reset_offsets_to_default(self) -> None:
        with self.lock:
            for servo in SERVOS:
                self.offsets[servo] = float(DEFAULT_OFFSETS.get(servo, 0.0))
        print("Offsets reset to defaults")

    def calibration(self) -> Dict[str, object]:
        with self.lock:
            return {
                "offsets": {str(s): float(self.offsets.get(s, 0.0))
                            for s in SERVOS},
                "inverted": sorted(self.inverted),
            }

    def positions(self) -> Dict[str, int]:
        """Фактическое (сглаженное) положение оси — для живого индикатора."""
        with self.lock:
            return {"0": int(round(self._move_positions.get(
                0, float(self.current_angles.get(0, 0)))))}

    def load_calibration(self, path: Path = CONFIG_PATH) -> bool:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return False
        try:
            offsets = data.get("offsets") or {}
            inverted = data.get("inverted")
            channels = set()
            for ch_str in offsets:
                try:
                    channels.add(int(ch_str))
                except (TypeError, ValueError):
                    continue
            if isinstance(inverted, list):
                try:
                    channels.update(int(c) for c in inverted)
                except (TypeError, ValueError):
                    pass
            if not channels <= set(SERVOS):
                # Файл от старой 12-канальной механики: offsets/инверсия
                # не имеют отношения к четырём сервам плеча.
                print(f"[Servo] {path}: калибровка старого робота — игнорирую")
                return False
            for ch_str, value in offsets.items():
                servo = int(ch_str)
                self.offsets[servo] = float(value)
            if isinstance(inverted, list):
                self.inverted = {int(c) for c in inverted if int(c) in SERVOS}
            print(f"[Servo] Калибровка загружена из {path}")
            return True
        except Exception as e:
            print(f"[Servo] Ошибка чтения калибровки: {e}")
            return False

    def save_calibration(self, path: Path = CONFIG_PATH) -> bool:
        try:
            payload = {
                "offsets": {str(s): float(self.offsets.get(s, 0.0))
                            for s in SERVOS},
                "inverted": sorted(self.inverted),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())        # обрыв питания не должен резать JSON
            print(f"[Servo] Калибровка сохранена в {path}")
            return True
        except Exception as e:
            print(f"[Servo] Не удалось сохранить калибровку: {e}")
            return False

    def set_inverted(self, channel: int, inverted: bool) -> bool:
        if channel not in SERVOS:
            return False
        with self.lock:
            if inverted:
                self.inverted.add(channel)
            else:
                self.inverted.discard(channel)
        return True

    # ------------------------------------------------------------------
    # Управление
    # ------------------------------------------------------------------
    def set_servo(self, channel: int, angle: int, smooth: bool = True) -> bool:
        """Двигать плечо логическим углом 0..90.

        ``smooth=True`` ведёт mover по трапеции скорости; ``smooth=False``
        ставит угол мгновенно.
        """
        if channel not in self.channel_configs:
            return False
        if not self.initialized or self.pwm is None:
            print(f"PCA9685 не инициализирована, канал {channel} не установлен")
            return False
        if not self._enabled:
            return False

        command_min, command_max = self.command_limits[channel]
        target_command = int(max(command_min, min(command_max, angle)))
        now = time.monotonic()

        if not smooth:
            with self.lock:
                self.current_angles[channel] = target_command
                self._move_positions.pop(channel, None)
                self._move_velocities.pop(channel, None)
                self._move_accels.pop(channel, None)
                self._move_spans.pop(channel, None)
                self._last_target_time[channel] = now
            with self._mover_cond:
                self._move_targets.pop(channel, None)
            return self._set_servo_immediate(channel, target_command)

        self._ensure_mover()
        with self.lock:
            if channel not in self._move_positions:
                self._move_positions[channel] = float(
                    self.current_angles.get(channel, target_command))
            start = self._move_positions[channel]
            self.current_angles[channel] = target_command
            # Одиночный переезд (панель): запоминаем длину хода, чтобы на
            # коротких 5–10° трапеция успела плавно разогнаться и затормозить.
            # Поток целей (телеоп/IK) — это не одиночный ход, не ограничиваем.
            if now - self._last_target_time.get(channel, 0.0) > 0.25:
                self._move_spans[channel] = abs(float(target_command) - start)
            else:
                self._move_spans.pop(channel, None)
            self._last_target_time[channel] = now
        with self._mover_cond:
            self._move_targets[channel] = target_command
            self._mover_cond.notify_all()
        return True

    def set_speed_limit(self, channels, deg_per_s) -> None:
        """Ограничить скорость канала, вирт. град/с. None — вернуть обычную."""
        with self.lock:
            for channel in channels:
                if deg_per_s is None:
                    self.speed_limits.pop(channel, None)
                else:
                    self.speed_limits[channel] = float(deg_per_s)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._mover_cond:
                if not self._move_targets:
                    return True
            time.sleep(0.02)
        return False

    def _ensure_mover(self) -> None:
        if not self._enabled:
            return
        with self.lock:
            thread = self._mover_thread
            if thread is not None and thread.is_alive():
                self._mover_stop.clear()
                return
            self._mover_stop.clear()
            mover = threading.Thread(
                target=self._mover_loop, daemon=True, name="servo-mover")
            self._mover_thread = mover
            mover.start()

    def _mover_loop(self) -> None:
        tick = MOVE_TICK
        while not self._mover_stop.is_set():
            if not self._enabled:
                break
            with self._mover_cond:
                if not self._move_targets:
                    self._mover_cond.wait(timeout=0.1)
                    if self._mover_stop.is_set():
                        break
                    if not self._move_targets:
                        continue
                channels = list(self._move_targets.keys())
            for channel in channels:
                try:
                    self._step_channel(channel, tick)
                except Exception as exc:               # noqa: BLE001
                    print(f"[Servo] mover: канал {channel}: {exc}", flush=True)
                    with self._mover_cond:
                        self._move_targets.pop(channel, None)
                    with self.lock:
                        self._move_velocities.pop(channel, None)
                        self._move_accels.pop(channel, None)
                        self._move_spans.pop(channel, None)
            time.sleep(tick)

    def _step_channel(self, channel: int, tick: float) -> None:
        with self._mover_cond:
            target = self._move_targets.get(channel)
        if target is None:
            return
        command_min, command_max = self.command_limits[channel]
        with self.lock:
            current = self._move_positions.get(
                channel, float(self.current_angles.get(channel, target)))
            velocity = self._move_velocities.get(channel, 0.0)
            accel_state = self._move_accels.get(channel, 0.0)

        distance = float(target) - current
        if abs(distance) <= MOVE_DEADBAND:
            self._set_servo_immediate(channel, target)
            with self.lock:
                self._move_positions[channel] = float(target)
                self._move_velocities[channel] = 0.0
                self._move_accels[channel] = 0.0
                self._move_spans.pop(channel, None)
            with self._mover_cond:
                self._move_targets.pop(channel, None)
            return

        limit = self.speed_limits.get(channel)
        v_max = min(MOVE_MAX_SPEED, limit if limit else MOVE_MAX_SPEED)
        accel = MOVE_ACCEL if not limit else min(MOVE_ACCEL, max(4.0 * limit, 60.0))
        # Тормозим от половины ускорения: запас на дискретный тик и рывок.
        v_brake = math.sqrt(2.0 * accel * MOVE_BRAKE_FRAC * abs(distance))
        v_target = min(v_max, v_brake)
        # Короткий одиночный ход: пик не выше, чем позволяет разогнаться и
        # затормозить на этой длине (треугольный профиль).
        span = self._move_spans.get(channel)
        if span:
            v_target = min(v_target,
                           math.sqrt(accel * MOVE_BRAKE_FRAC * span))
        v_desired = v_target if distance > 0 else -v_target

        # Ограничиваем не только ускорение, но и его производную (jerk):
        # старт и финиш получаются плавными, без удара в конце.
        a_desired = (v_desired - velocity) / tick
        a_desired = max(-accel, min(accel, a_desired))
        da = a_desired - accel_state
        da = max(-MOVE_JERK * tick, min(MOVE_JERK * tick, da))
        accel_state += da
        velocity += accel_state * tick
        # Не даём инерции разгона перелететь профиль: иначе на коротких
        # ходах скорость успевает уйти выше тормозной и в конце рывок.
        # При смене направления (знаки разные) не режем — там работает accel.
        if v_desired * velocity >= 0.0 and abs(velocity) > abs(v_desired):
            velocity = v_desired
            accel_state = 0.0
        step = velocity * tick

        if abs(step) >= abs(distance):
            position = float(target)
            velocity = 0.0
            accel_state = 0.0
        else:
            position = float(current) + step
            position = float(max(command_min, min(command_max, position)))

        self._set_servo_immediate(channel, position)
        with self.lock:
            self._move_positions[channel] = position
            self._move_velocities[channel] = velocity
            self._move_accels[channel] = accel_state

    def _set_servo_immediate(self, channel: int, virtual_angle: float) -> bool:
        if channel not in self.channel_configs:
            return False
        if self.pwm is None and not self._reinit_pwm():
            return False
        for attempt in range(4):
            if self._write_pulses(virtual_angle):
                return True
            time.sleep(0.01 * (attempt + 1))
        if self._reinit_pwm(min_interval=1.0) and self._write_pulses(virtual_angle):
            return True
        print(f"Ошибка установки сервоприводов: {self._last_error}")
        return False

    # ------------------------------------------------------------------
    # Расслабление
    # ------------------------------------------------------------------
    def relax_all(self) -> None:
        """Снять PWM со всех четырёх серв, чтобы они не держали нагрузку."""
        self._enabled = False
        self._mover_stop.set()
        with self._mover_cond:
            self._mover_cond.notify_all()
        with self.lock:
            self._move_velocities.clear()
            self._move_accels.clear()
            self._move_spans.clear()
        time.sleep(0.05)
        if not self.is_connected():
            self._reinit_pwm(min_interval=0.0, restore=False)
        if self.pwm is None:
            print("[Servo] relax_all: PCA9685 недоступна — сигнал снять нельзя")
            return
        relaxed = 0
        for servo in SERVOS:
            for attempt in range(3):
                if self._write_off(servo):
                    relaxed += 1
                    break
                self._reinit_pwm(min_interval=0.0, restore=False)
                time.sleep(0.01 * (attempt + 1))
        print(f"[Servo] Сервы расслаблены ({relaxed}/{len(SERVOS)})")

    def enable_all(self) -> None:
        """Разрешить управление сервами снова (снимает блок relax_all)."""
        self._enabled = True
        self._mover_stop.clear()


if __name__ == "__main__":
    # Самопроверка: логика 0/45/90 -> физ. 1/135/269 и зеркало 269/135/1.
    def us(physical: float) -> float:
        return (PULSE_MIN_US + (PULSE_MAX_US - PULSE_MIN_US)
                * physical / PHYSICAL_MAX)

    checks = ((0.0, us(1.0), us(269.0)),
              (45.0, us(135.0), us(135.0)),
              (90.0, us(269.0), us(1.0)))
    for v, normal, mirrored in checks:
        a = virtual_to_pulse_us(v, inverted=False)
        b = virtual_to_pulse_us(v, inverted=True)
        assert abs(a - normal) < 1e-6, (v, a, normal)
        assert abs(b - mirrored) < 1e-6, (v, b, mirrored)
    print(f"servo.py: 0..90° -> физ. 1..269° "
          f"({us(1.0):.0f}..{us(269.0):.0f} мкс) OK")
