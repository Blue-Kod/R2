#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пофайловый тест направлений рук на устройстве.

Запуск на роботе (в каталоге проекта):
    python3 -m robov_core.arm_test [--delta 30] [--pause 3] [--relax]

Скрипт поочерёдно двигает каждый сустав обеих рук на +delta° от позы
покоя и печатает, в какую сторону (по модели) должен пойти конец руки.
Сверьте физическое движение с ожиданием. Расхождения — это оси, которые
нужно чинить в INVERTED_CHANNELS / ARM_CHANNELS / to_servo_commands.

Оси (модель, система координат камеры):
  +Z = вперёд (от робота к человеку), +Y = вверх, +X = вправо от корпуса.
"""

import argparse
import sys
import time

import numpy as np

from robov_core import arm_kinematics
from robov_core.servo import ServoController, DEFAULT_POSE

SIDES = ("right", "left")
DELTA = 30.0
PAUSE = 3.0


def direction_words(delta: np.ndarray) -> str:
    words = []
    z, y, x = float(delta[2]), float(delta[1]), float(delta[0])
    th = 30.0  # мм — порог «заметно сдвинулся»
    if x > th:
        words.append("вправо от корпуса")
    elif x < -th:
        words.append("влево от корпуса")
    if y > th:
        words.append("вверх")
    elif y < -th:
        words.append("вниз")
    if z > th:
        words.append("вперёд (к человеку)")
    elif z < -th:
        words.append("назад")
    if not words:
        words.append("почти не двигается")
    return ", ".join(words)


def run(servo: ServoController, delta: float, pause: float) -> None:
    for ch, angle in DEFAULT_POSE.items():
        if ch in servo.channel_configs:
            servo.set_servo(ch, angle, smooth=True)
    time.sleep(pause)
    print("\n--- Поза покоя достигнута. Начинаю тест. ---")

    theta0 = (0.0, 0.0, 0.0)
    for side in SIDES:
        chans = arm_kinematics.ARM_CHANNELS[side]
        rest = arm_kinematics.rest_angles(side == "left")
        ee0 = arm_kinematics.fk(theta0, left=side == "left")["EE"]
        for joint in arm_kinematics.JOINT_NAMES:
            theta = list(theta0)
            idx = arm_kinematics.JOINT_NAMES.index(joint)
            theta[idx] += delta
            ee = arm_kinematics.fk(tuple(theta), left=side == "left")["EE"]
            commands = arm_kinematics.to_servo_commands(tuple(theta), left=side == "left")
            ch = chans[joint]
            expectation = direction_words(ee - ee0)
            print(f"\n[{side} / {joint}] канал ch{ch}, "
                  f"команда {rest[ch]} -> {commands[ch]}")
            print(f"  ОЖИДАНИЕ по модели: конец руки -> {expectation}")
            print(f"  Двигаю на {int(pause)} с...")
            servo.set_servo(ch, commands[ch], smooth=True)
            time.sleep(pause)
            servo.set_servo(ch, rest[ch], smooth=True)
            time.sleep(1.0)
            print(f"  -> вернул в rest ({rest[ch]}). ВЕРНО?")
        print(f"\n=== {side} arm done ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=float, default=DELTA,
                        help="величина пробного шага, градусы")
    parser.add_argument("--pause", type=float, default=PAUSE,
                        help="пауза в позе теста, сек")
    parser.add_argument("--relax", action="store_true",
                        help="расслабить сервы после теста")
    args = parser.parse_args()

    servo = ServoController(bus=0, address=0x40, freq=50)
    if not servo.initialized:
        print("ВНИМАНИЕ: PCA9685 не инициализирована (mock-режим), "
              "физического движения не будет.", file=sys.stderr)
    try:
        run(servo, args.delta, args.pause)
    except KeyboardInterrupt:
        print("\nПрервано.")
    finally:
        for ch, angle in DEFAULT_POSE.items():
            if ch in servo.channel_configs:
                servo.set_servo(ch, angle, smooth=True)
        if args.relax:
            servo.relax_all()
            print("Сервы расслаблены.")


if __name__ == "__main__":
    main()
