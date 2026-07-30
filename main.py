#!/usr/bin/env python3
"""R2 Robot - Main entry point."""

# -*- coding: utf-8 -*-

import argparse
import array
import os
import subprocess
import tempfile
import wave

from robov_core.high_level import start_background, command, speak, is_voice_active, APP_VERSION
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

        _wav_path = os.path.join(os.path.dirname(__file__), "robov_core", "stt", "sounds", "PowerOn.wav")
        if os.path.isfile(_wav_path):
            try:
                with wave.open(_wav_path, 'rb') as w:
                    params = w.getparams()
                    frames = bytearray(w.readframes(w.getnframes()))
                samples = array.array('h')
                samples.frombytes(bytes(frames))
                for i in range(len(samples)):
                    val = samples[i] * 2
                    if val > 32767: val = 32767
                    if val < -32768: val = -32768
                    samples[i] = val
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    with wave.open(tmp, 'wb') as w:
                        w.setparams(params)
                        w.writeframes(samples.tobytes())
                    tmp_path = tmp.name
                subprocess.run(
                    ["aplay", "-D", "plughw:1,0", tmp_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                os.unlink(tmp_path)
            except Exception:
                pass
        try:
            listener.input_cycle(soft_command, activation_check=is_voice_active)
        except KeyboardInterrupt:
            pass
        from robov_core.high_level import cleanup
        cleanup()
