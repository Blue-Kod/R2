#!/usr/bin/env python3
"""R2 Robot - Main entry point."""

import argparse

from robov_core.high_level import start_background, command, speak, APP_VERSION
from robov_core.stt import VoiceListener


def soft_command(text):
    """Handle simple phrases instantly, pass everything to AI."""
    if "привет" in text or "здравствуй" in text:
        speak("Привет!")
        return
    if "как дела" in text:
        speak("Хорошо, спасибо!")
        return
    command(text)


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
        listener = VoiceListener()
        try:
            listener.input_cycle(soft_command)
        except KeyboardInterrupt:
            pass
        from robov_core.high_level import cleanup
        cleanup()
