"""
inmp441 - Python driver for INMP441 I2S microphone via SPI3
Orange Pi 4 Pro (Allwinner A733)

Requires: libinmp441.so (built with make)
Optional: numpy (for array output)

Usage:
    import inmp441

    # Record 5 seconds
    audio = inmp441.record(seconds=5)

    # Record 10 seconds at 16kHz
    audio = inmp441.record(seconds=10, sample_rate=16000)

    # Streaming
    mic = inmp441.Microphone(sample_rate=8000)
    mic.start()
    data = mic.read(seconds=2)
    mic.stop()

    # Use with Whisper STT
    import whisper
    audio = inmp441.record(seconds=10, sample_rate=16000)
    audio_float = audio.astype(np.float32) / 32768.0
    result = whisper.transcribe(model, audio_float)
"""

import os
import time
import ctypes
import ctypes.util
from pathlib import Path

# ── Find shared library ────────────────────────────────────────

_LIB_SEARCH = [
    Path(__file__).parent / "libinmp441.so",
    Path(__file__).parent / "build" / "libinmp441.so",
    Path("/usr/local/lib/libinmp441.so"),
    Path("/usr/lib/libinmp441.so"),
]

_lib = None
for p in _LIB_SEARCH:
    if p.exists():
        _lib = ctypes.CDLL(str(p))
        break

if _lib is None:
    raise ImportError(
        "libinmp441.so not found. Build it with: make lib\n"
        f"Searched: {[str(p) for p in _LIB_SEARCH]}"
    )

# ── Bind C API ─────────────────────────────────────────────────

class _Config(ctypes.Structure):
    _fields_ = [
        ("sample_rate", ctypes.c_uint32),
        ("spi_speed",   ctypes.c_uint32),
        ("spi_device",  ctypes.c_char_p),
        ("gain",        ctypes.c_int),
    ]

_lib.inmp441_start.argtypes = [ctypes.POINTER(_Config)]
_lib.inmp441_start.restype = ctypes.c_void_p

_lib.inmp441_read.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_uint32]
_lib.inmp441_read.restype = ctypes.c_int

_lib.inmp441_read_blocking.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16),
                                        ctypes.c_uint32, ctypes.c_uint32]
_lib.inmp441_read_blocking.restype = ctypes.c_int

_lib.inmp441_stop.argtypes = [ctypes.c_void_p]
_lib.inmp441_stop.restype = None

_lib.inmp441_available.argtypes = [ctypes.c_void_p]
_lib.inmp441_available.restype = ctypes.c_uint64


# ── Public API ─────────────────────────────────────────────────

class Microphone:
    """Streaming microphone interface.

    Args:
        sample_rate: Audio sample rate in Hz (default: 8000)
        spi_device:  SPI device path (default: /dev/spidev3.0)
        spi_speed:   SPI clock speed in Hz (0 = auto)
        gain:        Gain multiplier (default: 64)

    Requires root (sudo) for GPIO access.
    """

    def __init__(self, sample_rate=8000, spi_device="/dev/spidev3.0", spi_speed=4000000,
                 gain=64):
        self.sample_rate = sample_rate
        self.spi_device = spi_device.encode() if isinstance(spi_device, str) else spi_device
        self.spi_speed = spi_speed
        self.gain = gain
        self._handle = None

    def start(self):
        """Start capturing audio from the microphone."""
        if self._handle is not None:
            return
        cfg = _Config(
            sample_rate=self.sample_rate,
            spi_speed=self.spi_speed,
            spi_device=self.spi_device,
            gain=self.gain,
        )
        self._handle = _lib.inmp441_start(ctypes.byref(cfg))
        if not self._handle:
            raise RuntimeError("Failed to start INMP441 capture (need sudo?)")

    def stop(self):
        """Stop capturing and release resources."""
        if self._handle is not None:
            _lib.inmp441_stop(self._handle)
            self._handle = None

    def read(self, seconds=1.0, blocking=True):
        """Read audio data.

        Args:
            seconds:   Duration in seconds to read
            blocking:  If True, block until data available. If False, return
                       whatever is available (may be empty).

        Returns:
            numpy.ndarray (int16) if numpy is available, else bytes.
        """
        num_samples = int(self.sample_rate * seconds)
        buf = (ctypes.c_int16 * num_samples)()

        if blocking:
            n = _lib.inmp441_read_blocking(self._handle, buf, num_samples, 0)
        else:
            n = _lib.inmp441_read(self._handle, buf, num_samples)

        if n <= 0:
            return self._empty(seconds)

        try:
            import numpy as np
            return np.ctypeslib.as_array(buf[:n]).astype(np.int16).copy()
        except ImportError:
            return bytes(buf[:n])

    def read_samples(self, num_samples, blocking=True):
        """Read exact number of samples.

        Args:
            num_samples: Number of int16 samples to read
            blocking:    If True, block until all samples available

        Returns:
            numpy.ndarray (int16) or bytes
        """
        buf = (ctypes.c_int16 * num_samples)()
        if blocking:
            n = _lib.inmp441_read_blocking(self._handle, buf, num_samples, 0)
        else:
            n = _lib.inmp441_read(self._handle, buf, num_samples)

        if n <= 0:
            try:
                import numpy as np
                return np.zeros(num_samples, dtype=np.int16)
            except ImportError:
                return b'\x00' * (num_samples * 2)

        try:
            import numpy as np
            return np.ctypeslib.as_array(buf[:n]).astype(np.int16).copy()
        except ImportError:
            return bytes(buf[:n])

    @property
    def available(self):
        """Number of samples available to read without blocking."""
        return _lib.inmp441_available(self._handle) if self._handle else 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()

    def _empty(self, seconds):
        try:
            import numpy as np
            return np.zeros(int(self.sample_rate * seconds), dtype=np.int16)
        except ImportError:
            return b'\x00' * int(self.sample_rate * seconds * 2)


def record(seconds=5, sample_rate=8000, spi_device="/dev/spidev3.0", spi_speed=4000000,
           gain=64):
    """Record audio and return as numpy array.

    Args:
        seconds:     Duration in seconds
        sample_rate: Sample rate in Hz
        spi_device:  SPI device path
        spi_speed:   SPI clock speed in Hz (0 = auto)
        gain:        Gain multiplier (default: 64)

    Returns:
        numpy.ndarray of int16 samples
    """
    with Microphone(sample_rate=sample_rate, spi_device=spi_device, spi_speed=spi_speed,
                    gain=gain) as mic:
        return mic.read(seconds=seconds)


def record_to_file(filename, seconds=5, sample_rate=8000, spi_device="/dev/spidev3.0", spi_speed=4000000,
                   gain=64):
    """Record and save as WAV file.

    Args:
        filename:    Output .wav path
        seconds:     Duration in seconds
        sample_rate: Sample rate in Hz
        spi_device:  SPI device path
        spi_speed:   SPI clock speed in Hz (0 = auto)
        gain:        Gain multiplier (default: 64)

    Returns:
        Number of samples written
    """
    audio = record(seconds=seconds, sample_rate=sample_rate, spi_device=spi_device, spi_speed=spi_speed,
                   gain=gain)
    _write_wav(filename, audio, sample_rate)
    return len(audio)


def _write_wav(filename, data, sample_rate):
    """Write int16 data as WAV file (no external deps)."""
    import struct
    n = len(data)
    data_bytes = n * 2
    with open(filename, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', data_bytes + 36))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate,
                            sample_rate * 2, 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_bytes))
        if hasattr(data, 'tobytes'):
            f.write(data.tobytes())
        else:
            f.write(data)


# ── Convenience: quick test ────────────────────────────────────

if __name__ == "__main__":
    import sys

    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rate = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    print(f"Recording {seconds}s at {rate} Hz...")
    audio = record(seconds=seconds, sample_rate=rate)

    try:
        import numpy as np
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        print(f"Peak: {peak} / 32767  RMS: {rms:.0f}")
    except ImportError:
        print(f"Recorded {len(audio)} bytes")

    out = "test_recording.wav"
    _write_wav(out, audio, rate)
    print(f"Saved: {out}")
