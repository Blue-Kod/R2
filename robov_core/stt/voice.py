"""Speech-to-text module: Silero VAD + OnlineRecognizer with context buffer."""

import os
import sys
import re
import time
import signal
import threading
import subprocess
import urllib.request
import numpy as np
import sherpa_onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_R2 = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import inmp441

_listener = None

# ── Config ──────────────────────────────────────────────────────

SAMPLE_RATE = 16000
SPI_DEVICE = "/dev/spidev3.0"
SPI_SPEED = 4000000
GAIN = 64

MODEL_DIR = os.path.join(_R2, "models", "vosk-model-small-streaming-ru")
VAD_MODEL = os.path.join(_R2, "models", "silero_vad.onnx")
CHUNK_MS = 200
NUM_THREADS = 2

VAD_THRESHOLD = 0.3
VAD_MIN_SILENCE = 0.3
VAD_MIN_SPEECH = 0.1
CONTEXT_SECONDS = 1

RULE1_SILENCE = 2.0
RULE2_SILENCE = 1.0
RULE3_MAX_LEN = 300

SOUND_DIR = os.path.join(_HERE, "sounds")
SOUND_OFF = os.path.join(SOUND_DIR, "VoiceEnd.wav")
APLAY_DEV = "plughw:1,0"

TRIGGER_WORDS = ["R2", "два", "робот", "работа", "работай"]

HP_ALPHA = np.exp(-2 * np.pi * 150 / SAMPLE_RATE)
_HP_POWERS = None
_HP_INV_POWERS = None


def _highpass(x, zi):
    """Vectorized 1st-order high-pass: y[n] = x[n]-x[n-1] + alpha*y[n-1]."""
    global _HP_POWERS, _HP_INV_POWERS
    n = len(x)
    if _HP_POWERS is None or len(_HP_POWERS) < n:
        idx = np.arange(n, dtype=np.float64)
        _HP_POWERS = (HP_ALPHA ** idx)
        _HP_INV_POWERS = (HP_ALPHA ** (-idx))
    p = _HP_POWERS[:n]
    ip = _HP_INV_POWERS[:n]
    xf = x.astype(np.float64)
    d = np.empty(n, dtype=np.float64)
    d[0] = xf[0] - float(zi[0])
    d[1:] = xf[1:] - xf[:-1]
    y = np.cumsum(d * ip) * p
    return y.astype(np.float32), np.array([xf[-1]], dtype=np.float32)

_HF_ASR_BASE = "https://huggingface.co/alphacep/vosk-model-small-streaming-ru/resolve/main"
_ASR_FILES = {
    "am-onnx/encoder.int8.onnx": f"{_HF_ASR_BASE}/am-onnx/encoder.int8.onnx",
    "am-onnx/decoder.int8.onnx": f"{_HF_ASR_BASE}/am-onnx/decoder.int8.onnx",
    "am-onnx/joiner.int8.onnx":  f"{_HF_ASR_BASE}/am-onnx/joiner.int8.onnx",
    "lang/tokens.txt":            f"{_HF_ASR_BASE}/lang/tokens.txt",
}
_HF_VAD_URL = "https://huggingface.co/snakers4/silero-models/resolve/main/vad/silero_vad.onnx"


def _ensure_models():
    """Download model files if missing."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VAD_MODEL), exist_ok=True)

    for rel, url in _ASR_FILES.items():
        dest = os.path.join(MODEL_DIR, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"  ↓ {rel}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"({os.path.getsize(dest) // 1024}KB)", flush=True)

    if not os.path.exists(VAD_MODEL) or os.path.getsize(VAD_MODEL) < 100000:
        print("  ↓ silero_vad.onnx...", end=" ", flush=True)
        urllib.request.urlretrieve(_HF_VAD_URL, VAD_MODEL)
        print(f"({os.path.getsize(VAD_MODEL) // 1024}KB)", flush=True)


def _load_recognizer():
    """Load the ASR recognizer (slow on ARM, ~5-10s)."""
    t0 = time.monotonic()
    enc = os.path.join(MODEL_DIR, "am-onnx/encoder.int8.onnx")
    dec = os.path.join(MODEL_DIR, "am-onnx/decoder.int8.onnx")
    jnr = os.path.join(MODEL_DIR, "am-onnx/joiner.int8.onnx")
    tok = os.path.join(MODEL_DIR, "lang/tokens.txt")
    print("  encoder...", end=" ", flush=True)
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=enc,
        decoder=dec,
        joiner=jnr,
        tokens=tok,
        num_threads=NUM_THREADS,
        sample_rate=SAMPLE_RATE,
        dither=3e-5,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=RULE1_SILENCE,
        rule2_min_trailing_silence=RULE2_SILENCE,
        rule3_min_utterance_length=RULE3_MAX_LEN,
        decoding_method="modified_beam_search",
        max_active_paths=20,
        hotwords_file=os.path.join(_HERE, "hotwords_ru.txt"),
        hotwords_score=5.0,
    )
    print(f"done ({time.monotonic() - t0:.1f}s)", flush=True)
    return rec


def _play_sound_blocking(path, mic=None):
    if not os.path.isfile(path):
        return
    subprocess.run(
        ["aplay", "-D", APLAY_DEV, path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if mic is not None:
        while mic.available > 0:
            mic.read_samples(min(mic.available, 6400), blocking=False)


def _fix_text(text):
    text = re.sub(r'\bртва\b', 'R2', text)
    text = re.sub(r'\bмертвак\b', 'R2', text)
    text = re.sub(r'\bрд\b', 'R2', text)
    text = re.sub(r'\bэрд\b', 'R2', text)
    text = re.sub(r'\bэр\s*два\b', 'R2', text)
    text = re.sub(r'\bр\s*д\s*ва\b', 'R2', text)
    text = re.sub(r'\bр\s*2\b', 'R2', text)
    text = re.sub(r'\bр\s*\.\s*2\b', 'R2', text)
    return text


class VoiceListener:
    """Streaming voice recognition with Silero VAD.

    Usage:
        listener = VoiceListener()
        listener.start()
        ...
        listener.stop()
    """

    def __init__(self):
        global _listener
        self._handler = print
        self._activation_check = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread = None
        _listener = self

        _ensure_models()
        self.recognizer = _load_recognizer()

        silero_cfg = sherpa_onnx.SileroVadModelConfig(
            model=VAD_MODEL,
            threshold=VAD_THRESHOLD,
            min_silence_duration=VAD_MIN_SILENCE,
            min_speech_duration=VAD_MIN_SPEECH,
            max_speech_duration=30,
        )
        self.vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=silero_cfg,
            sample_rate=SAMPLE_RATE,
            num_threads=NUM_THREADS,
        )

        self.mic = inmp441.Microphone(
            sample_rate=SAMPLE_RATE,
            spi_device=SPI_DEVICE,
            spi_speed=SPI_SPEED,
            gain=GAIN,
        )
        print("voice init completed", flush=True)

    def pause(self):
        self._paused.set()

    def unpause(self):
        while self.mic.available > 0:
            self.mic.read_samples(min(self.mic.available, 6400), blocking=False)
        time.sleep(0.1)
        self._paused.clear()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def input_cycle(self, function=None, activation_check=None):
        """Block and process voice input in a loop (Ctrl+C to stop)."""
        if function is not None:
            self._handler = function
        if activation_check is not None:
            self._activation_check = activation_check
        try:
            signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        except ValueError:
            pass
        self._loop()

    def run(self):
        try:
            signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        except ValueError:
            pass
        self._loop()

    def _loop(self):
        vad = sherpa_onnx.VoiceActivityDetector(
            self.vad_config, buffer_size_in_seconds=60
        )
        self.mic.start()
        time.sleep(0.5)

        chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
        hp_zi = np.zeros(1, dtype=np.float32)

        ctx_size = SAMPLE_RATE * CONTEXT_SECONDS
        ctx_buf = np.zeros(ctx_size, dtype=np.float32)
        ctx_pos = 0
        ctx_count = 0

        rec_stream = None
        state = "SLEEP"
        command_text = ""

        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    self.mic.read_samples(chunk_samples, blocking=True)
                    continue

                raw = self.mic.read_samples(chunk_samples, blocking=True)
                if isinstance(raw, bytes):
                    raw = np.frombuffer(raw, dtype=np.int16)
                samples = raw.astype(np.float32) / 32768.0

                rms = float(np.sqrt(np.mean(samples ** 2)) * 32768)
                if rms < 10:
                    continue

                audio, hp_zi = _highpass(samples, hp_zi)
                np.tanh(audio, out=audio)
                audio *= 0.95
                audio_f32 = np.clip(audio, -1.0, 1.0).astype(np.float32)

                n = len(audio_f32)
                end = min(ctx_pos + n, ctx_size)
                fit = end - ctx_pos
                ctx_buf[ctx_pos:end] = audio_f32[:fit]
                if n > fit:
                    ctx_buf[: n - fit] = audio_f32[fit:]
                ctx_pos = (ctx_pos + n) % ctx_size
                ctx_count += n

                vad.accept_waveform(audio_f32)

                if state == "SLEEP":
                    if vad.is_speech_detected():
                        state = "ACTIVE"
                        command_text = ""
                        if ctx_count >= ctx_size:
                            ctx_full = np.concatenate([
                                ctx_buf[ctx_pos:],
                                ctx_buf[:ctx_pos],
                            ])
                        else:
                            ctx_full = ctx_buf[:ctx_count].copy()

                        rec_stream = self.recognizer.create_stream()
                        rec_stream.accept_waveform(SAMPLE_RATE, ctx_full)
                        while self.recognizer.is_ready(rec_stream):
                            self.recognizer.decode_stream(rec_stream)
                    continue

                if rec_stream is not None:
                    rec_stream.accept_waveform(SAMPLE_RATE, audio_f32)
                    while self.recognizer.is_ready(rec_stream):
                        self.recognizer.decode_stream(rec_stream)

                    result = self.recognizer.get_result(rec_stream).strip().lower()
                    if result:
                        command_text = _fix_text(result)

                    if not vad.is_speech_detected():
                        final = _fix_text(command_text.strip())
                        if final:
                            print(f"voice text {final}", flush=True)
                            if self._activation_check is None or self._activation_check(final):
                                if any(w in final for w in TRIGGER_WORDS):
                                    _play_sound_blocking(SOUND_OFF, self.mic)
                                self._handler(final)
                        state = "SLEEP"
                        command_text = ""
                        rec_stream = None

        finally:
            self.mic.stop()


if __name__ == "__main__":
    def handler(text):
        print(f"  >> {text}")

    listener = VoiceListener()
    listener.run()