#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from ai import command, send_frame
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
        command("Система запущена. Скажи 'Здравствуйте. Я готов к работе.'")
        while True:
            time.sleep(1)
            send_frame()