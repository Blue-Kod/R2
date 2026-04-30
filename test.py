#!/usr/bin/env python3
import sys, os, time, ctypes
import numpy as np

# ========== 1. Полное подавление ALSA (оставляем только print) ==========
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)

import speech_recognition as sr

# ========== 2. Попытка подключить нейросетевой шумодав RNNoise ==========
try:
    import rnnoise_python as rn
    USE_RNNOISE = True
    print("[+] RNNoise (нейросетевой шумодав) загружен")
except ImportError:
    import noisereduce as nr
    USE_RNNOISE = False
    print("[!] RNNoise не найден, используется noisereduce")

# ========== 3. Настройки ==========
SAMPLE_RATE = 48000
AGC_ENABLE = True               # автоматическая регулировка усиления
AGC_TARGET_LEVEL = -20.0        # целевой уровень RMS в dB (чтобы тихий голос поднимался)
AGC_MAX_GAIN_DB = 20.0          # максимальное усиление, дБ
AGC_DECAY_RATE = 0.01           # скорость плавного спада усиления

# Для noisereduce (если не RNNoise)
PROP_DECREASE = 0.7
noise_profile = None            # для noisereduce (не используется в RNNoise)

# Глобальная переменная для AGC
_agc_gain_linear = 1.0

def rms_db(samples: np.ndarray) -> float:
    """Среднеквадратичное значение в децибелах."""
    rms = np.sqrt(np.mean(samples.astype(np.float64)**2))
    return 20.0 * np.log10(max(rms, 1e-10))

def apply_agc(audio: np.ndarray) -> np.ndarray:
    """
    Простой AGC: плавно подстраивает усиление, чтобы уровень был близок к целевому.
    audio – float32 в диапазоне [-1, 1].
    """
    global _agc_gain_linear
    if not AGC_ENABLE:
        return audio

    current_rms_db = rms_db(audio)
    # Разница между целевым и текущим уровнем
    diff_db = AGC_TARGET_LEVEL - current_rms_db
    # Ограничиваем усиление
    diff_db = max(-AGC_MAX_GAIN_DB, min(AGC_MAX_GAIN_DB, diff_db))
    target_gain_linear = 10 ** (diff_db / 20.0)

    # Плавное изменение (decay)
    _agc_gain_linear = _agc_gain_linear + AGC_DECAY_RATE * (target_gain_linear - _agc_gain_linear)

    amplified = audio * _agc_gain_linear
    # Защита от клиппинга
    amplified = np.clip(amplified, -1.0, 1.0)
    return amplified

def process_with_rnnoise(audio_data: sr.AudioData) -> sr.AudioData:
    """Обработка аудио нейросетевым шумодавом RNNoise."""
    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

    # AGC усиление
    audio_np = apply_agc(audio_np)

    denoiser = rn.RNNoise()
    frame_size = 480  # обязательно для 48 кГц
    output = np.zeros_like(audio_np)
    for i in range(0, len(audio_np) - frame_size + 1, frame_size):
        frame = audio_np[i:i+frame_size].copy()
        out_frame = denoiser.process_frame(frame)
        output[i:i+frame_size] = out_frame
    # Последний неполный кадр можно оставить без обработки
    output_int16 = (np.clip(output, -1, 1) * 32767).astype(np.int16)
    return sr.AudioData(output_int16.tobytes(), SAMPLE_RATE, 2)

def process_with_noisereduce(audio_data: sr.AudioData) -> sr.AudioData:
    """Обработка через классический noisereduce + AGC."""
    global noise_profile
    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

    # AGC усиление
    audio_np = apply_agc(audio_np)

    if noise_profile is not None:
        audio_np = nr.reduce_noise(
            y=audio_np,
            sr=SAMPLE_RATE,
            y_noise=noise_profile,
            prop_decrease=PROP_DECREASE,
            n_fft=1024,
            win_length=1024,
            hop_length=512
        )
    reduced_int16 = (np.clip(audio_np, -1, 1) * 32767).astype(np.int16)
    return sr.AudioData(reduced_int16.tobytes(), SAMPLE_RATE, 2)

def capture_noise_profile(recognizer, mic):
    """Захват образца шума (только для noisereduce, не для rnnoise)."""
    global noise_profile
    if USE_RNNOISE:
        print("[i] RNNoise не требует профиля шума, пропускаем")
        return
    print(f"[*] Запись образца шума (2 сек, молчите)...")
    with mic as source:
        audio_sample = recognizer.record(source, duration=2)
    audio_np = np.frombuffer(audio_sample.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
    noise_profile = audio_np
    print("[+] Профиль шума сохранён")

def callback(recognizer, audio):
    """Фоновый обработчик каждого сказанного куска."""
    try:
        if USE_RNNOISE:
            cleaned = process_with_rnnoise(audio)
        else:
            cleaned = process_with_noisereduce(audio)

        text = recognizer.recognize_google(cleaned, language="ru-RU")
        print(f"User said: {text}")
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")

def main():
    r = sr.Recognizer()
    # Чувствительность детектора речи подстраиваем под ослабленный шумодавом шум
    r.dynamic_energy_threshold = True
    r.energy_threshold = 4000      # начальное значение
    r.pause_threshold = 1.2
    r.phrase_threshold = 0.15

    mic = sr.Microphone(device_index=1, sample_rate=SAMPLE_RATE)

    # Шаг 1: образец шума (если нужно)
    capture_noise_profile(r, mic)

    # Шаг 2: калибровка фона
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
        # После калибровки порог всё равно может быть высоким из-за вентилятора.
        # Снижаем его, потому что шумодав уберёт фон, и нам не нужно бояться ложных срабатываний.
        r.energy_threshold = max(r.energy_threshold * 0.3, 80)
        print(f"[*] Порог энергии: {r.energy_threshold:.1f}")

    print("[*] Слушаю... (нажмите Enter для выхода)")
    stop = r.listen_in_background(mic, callback)
    input()
    stop(wait_for_stop=True)
    print("[*] Микрофон освобождён.")

if __name__ == "__main__":
    main()