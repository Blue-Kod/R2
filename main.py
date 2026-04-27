#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
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

APP_VERSION = "0.1"
running = True
EMOTE_CYCLE = [os.path.splitext(f)[0] for f in os.listdir("emotions") if f.endswith(".png")]
_emote_index = 0

def parse_args():
    parser = argparse.ArgumentParser(description="R2 launcher")
    parser.add_argument("--version", action="store_true", help="Show version")
    return parser.parse_args()


def main_loop():
    global _emote_index
    emotion = EMOTE_CYCLE[_emote_index]
    print(emotion)
    emote(emotion)
    _emote_index = (_emote_index + 1) % len(EMOTE_CYCLE)
    time.sleep(3)


if __name__ == "__main__":
    args = parse_args()
    if args.version:
        print(f"R2 v{APP_VERSION}")
    else:
        start_background()
        # Main loop now runs in current thread, emote updates are pushed to display
        while running:
            main_loop()