#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from ai import command, start_voice_mode
from r2_app.high_level import *

APP_VERSION = "0.1"
running = True
EMOTE_CYCLE = [os.path.splitext(f)[0] for f in os.listdir("emotions") if f.endswith(".png")]
_emote_index = 0

def parse_args():
    parser = argparse.ArgumentParser(description="R2 launcher")
    parser.add_argument("--version", action="store_true", help="Show version")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.version:
        print(f"R2 v{APP_VERSION}")
    else:
        start_background()
        # Первый вызов command() – при AUDIO_INPUT=True он перейдёт в бесконечный голосовой режим
        command("Система запущена. Скажи 'Здравствуйте. Я готов к работе.'")
        # Если голосовой ввод выключен (режим по умолчанию), просто держим процесс
        while True:
            time.sleep(3)