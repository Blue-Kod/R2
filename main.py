#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time

from robov_core import ai
from robov_core.ai import command, send_frame
from robov_core.high_level import *

APP_VERSION = "0.1"
running = True
EMOTE_CYCLE = [os.path.splitext(f)[0] for f in os.listdir("robov_core/emotions") if f.endswith(".png")]
_emote_index = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Robov-core launcher")
    parser.add_argument("--version", action="store_true", help="Show version")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.version:
        print(f"Robov-core v{APP_VERSION}")
    else:
        start_background()
        ai.command("Система запущена. Скажи 'Здравствуйте. Я готов к работе.'")
        while True:
            time.sleep(1)
            send_frame()