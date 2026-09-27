#!/usr/bin/env python3
"""R2 Robot - Main entry point."""

# -*- coding: utf-8 -*-

import argparse
import signal
import time

from robov_core.high_level import start_background, APP_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description="R2 Robot Controller")
    parser.add_argument("--version", action="store_true", help="Show version")
    return parser.parse_args()


def _handle_termination(signum, frame):
    """Systemd stop/reboot присылает SIGTERM — расслабляем сервы и выходим.

    Без этого при системном ребуте (не через наш /api/reboot) процесс
    умирает молча, и сервы остаются под нагрузкой.
    """
    try:
        from robov_core.high_level import cleanup
        cleanup()
    finally:
        raise SystemExit(0)


if __name__ == "__main__":
    args = parse_args()
    if args.version:
        print(f"R2 Robot v{APP_VERSION}")
    else:
        signal.signal(signal.SIGTERM, _handle_termination)
        start_background()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        from robov_core.high_level import cleanup
        cleanup()
