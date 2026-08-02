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
_HF_VAD_URL = "https://raw.githubusercontent.com/Blue-Kod/prepared-models/main/silero_vad.onnx"


def _ensure_models(asr_needed: bool = True):
    """Download model files if missing.

    asr_needed=False skips the (large) ASR encoder/decoder/joiner downloads;
    silero_vad is always downloaded since it drives speech detection.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VAD_MODEL), exist_ok=True)

    if asr_needed:
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


# ── Duck.ai dictation (online STT) ─────────────────────────────────

_DUCK_DICTATION_URL = "https://duck.ai/duckchat/v1/dictation"
_DUCK_MIN_UTTERANCE = 0.3  # seconds of human speech before uploading


def _get_duck_provider():
    """Lazily instantiate the everyllm DuckAI provider (reuses its VQD solver)."""
    global _duck_provider
    if _duck_provider is None:
        try:
            from robov_core.everyllm.providers import DuckAIProvider
        except ImportError:
            from ..everyllm.providers import DuckAIProvider
        _duck_provider = DuckAIProvider()
    return _duck_provider


_duck_provider = None


def _mp3_from_int16(pcm: np.ndarray) -> bytes:
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(2)
    return enc.encode(pcm.tobytes()) + enc.flush()


def _dictation_headers(vqd: str) -> dict:
    return {
        "accept": "text/event-stream",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "audio/mpeg",
        "pragma": "no-cache",
        "priority": "u=0",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        ),
        "Referer": "https://duck.ai/",
        "x-vqd-hash-1": vqd,
    }


def _dictate(mp3_bytes: bytes) -> str:
    """Transcribe an MP3 clip via duck.ai /dictation (blocks)."""
    import time

    import requests

    last = None
    delays = [1.0, 3.0, 7.0, 15.0]
    for attempt, delay in enumerate(delays + [0.0]):
        try:
            vqd = _get_duck_provider()._get_vqd()
            r = requests.post(
                _DUCK_DICTATION_URL,
                data=mp3_bytes,
                headers=_dictation_headers(vqd),
                timeout=60,
            )
            if r.status_code == 200:
                return r.json().get("text", "")
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        if delay:
            time.sleep(delay)
    raise RuntimeError(f"dictation failed: {last}")


class VoiceListener:
    """Streaming voice recognition with Silero VAD.

    Usage:
        listener = VoiceListener()
        listener.start()
        ...
        listener.stop()
    """

    def __init__(self, use_offline_stt: bool = False):
        global _listener
        self.use_offline_stt = use_offline_stt
        self._handler = print
        self._activation_check = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread = None
        _listener = self

        _ensure_models(asr_needed=self.use_offline_stt)
        self.recognizer = None
        if self.use_offline_stt:
            self._ensure_recognizer()

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

    def _ensure_recognizer(self):
        if self.recognizer is not None:
            return
        _ensure_models(asr_needed=True)
        self.recognizer = _load_recognizer()

    def _fallback_to_offline(self):
        if self.use_offline_stt:
            return
        print("duck.ai dictation failed — switching to offline ASR", flush=True)
        try:
            self._ensure_recognizer()
            self.use_offline_stt = True
        except Exception as e:  # noqa: BLE001
            print(f"offline ASR unavailable, staying online: {e}", flush=True)

    def _handle_online_utterance(self, raw_chunks):
        """Transcribe a VAD-cut speech clip via duck.ai and dispatch the text."""
        if not raw_chunks:
            return
        pcm = np.concatenate(raw_chunks)
        if len(pcm) < int(SAMPLE_RATE * _DUCK_MIN_UTTERANCE):
            return
        try:
            mp3 = _mp3_from_int16(pcm)
            text = _dictate(mp3)
        except Exception as e:  # noqa: BLE001
            print(f"dictation error: {e}", flush=True)
            self._fallback_to_offline()
            return
        final = _fix_text(text.strip())
        if final:
            print(f"voice text {final}", flush=True)
            if self._activation_check is None or self._activation_check(final):
                if any(w in final for w in TRIGGER_WORDS):
                    _play_sound_blocking(SOUND_OFF, self.mic)
                self._handler(final)

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
        utterance_raw = []

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
                        utterance_raw = []
                        if not self.use_offline_stt:
                            utterance_raw.append(raw)
                        if self.use_offline_stt:
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

                if not self.use_offline_stt:
                    utterance_raw.append(raw)
                if rec_stream is not None:
                    rec_stream.accept_waveform(SAMPLE_RATE, audio_f32)
                    while self.recognizer.is_ready(rec_stream):
                        self.recognizer.decode_stream(rec_stream)

                    result = self.recognizer.get_result(rec_stream).strip().lower()
                    if result:
                        command_text = _fix_text(result)

                if not vad.is_speech_detected():
                    try:
                        if self.use_offline_stt:
                            final = _fix_text(command_text.strip())
                            if final:
                                print(f"voice text {final}", flush=True)
                                if self._activation_check is None or self._activation_check(final):
                                    if any(w in final for w in TRIGGER_WORDS):
                                        _play_sound_blocking(SOUND_OFF, self.mic)
                                    self._handler(final)
                        else:
                            self._handle_online_utterance(utterance_raw)
                    except Exception as e:  # noqa: BLE001
                        print(f"STT dispatch error: {e}", flush=True)
                    state = "SLEEP"
                    command_text = ""
                    utterance_raw = []
                    rec_stream = None

        finally:
            self.mic.stop()


if __name__ == "__main__":
    def handler(text):
        print(f"  >> {text}")

    listener = VoiceListener()
    listener.run()