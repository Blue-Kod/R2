#!/usr/bin/env python3
"""R2 Robot - Main entry point."""

import argparse
import time

from robov_core.high_level import start_background, APP_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description="R2 Robot Controller")
    parser.add_argument("--version", action="store_true", help="Show version")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.version:
        print(f"R2 Robot v{APP_VERSION}")
    else:
        start_background()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            from robov_core.high_level import cleanup
            cleanup()
