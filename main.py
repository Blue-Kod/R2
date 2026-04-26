#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from r2_app.high_level import (
    angle,
    build_point_cloud,
    emote,
    gemini_test,
    get_camera,
    get_coords_stereo,
    get_stereo_camera,
    set_eyes_position,
    set_servo_tracking,
    start_background,
)

APP_VERSION = "2.0"
running = True
EMOTE_CYCLE = ["normal", "sad", "excited", "spooked", "unamused", "worried", "woozy", "angry", "wince"]
_emote_index = 0

def parse_args():
    parser = argparse.ArgumentParser(description="R2 launcher")
    parser.add_argument("--version", action="store_true", help="Показать версию")
    return parser.parse_args()


def main_loop():
    global _emote_index
    emotion = EMOTE_CYCLE[_emote_index]
    emote(emotion)
    _emote_index = (_emote_index + 1) % len(EMOTE_CYCLE)
    time.sleep(3)


if __name__ == "__main__":
    args = parse_args()
    if args.version:
        print(f"R2 v{APP_VERSION}")
    else:
        start_background()
        while running:
            main_loop()