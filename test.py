#!/usr/bin/env python3
"""
Распознавание речи с нейросетевым шумоподавлением (pyrnnoise)
и автоматическим усилением для дальнего источника.
"""
import sys
import os
import numpy as np

# 1. Полное отключение ALSA-сообщений
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)

import speech_recognition as sr

# 2. Подключаем pyrnnoise (должен быть установлен: pip install pyrnnoise)
try:
    from pyrnnoise import RNNoise
except ImportError:
    sys.stderr.write("Ошибка: требуется библиотека pyrnnoise. Установите: pip install pyrnnoise\n")
    sys.exit(1)

# ------------------ Настройки ------------------
SAMPLE_RATE = 48000

# AGC (автоматическая регулировка усиления)
AGC_ENABLE = True
AGC_TARGET_LEVEL = -20.0      # целевой уровень RMS в dB (тихий голос будет усилен)
AGC_MAX_GAIN_DB = 20.0        # максимальное усиление, дБ
AGC_DECAY_RATE = 0.01         # скорость адаптации (чем меньше, тем плавнее)

# Глобальное состояние для AGC
_agc_gain_linear = 1.0

# Инициализация нейросетевого шумоподавителя
denoiser = RNNoise(sample_rate=SAMPLE_RATE)

# -------------------------------------------------

def rms_db(samples: np.ndarray) -> float:
    """RMS уровень в децибелах (для float32 в диапазоне -1..1)."""
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    return 20.0 * np.log10(max(rms, 1e-10))

def apply_agc(audio: np.ndarray) -> np.ndarray:
    """
    Плавно поднимает уровень до AGC_TARGET_LEVEL.
    audio: float32 array, range [-1, 1].
    Возвращает усиленный float32 с защитой от клиппинга.
    """
    global _agc_gain_linear
    if not AGC_ENABLE:
        return audio

    current_db = rms_db(audio)
    diff_db = AGC_TARGET_LEVEL - current_db
    diff_db = max(-AGC_MAX_GAIN_DB, min(AGC_MAX_GAIN_DB, diff_db))
    target_gain_linear = 10.0 ** (diff_db / 20.0)

    # Экспоненциальное сглаживание усиления
    _agc_gain_linear += AGC_DECAY_RATE * (target_gain_linear - _agc_gain_linear)

    amplified = audio * _agc_gain_linear
    return np.clip(amplified, -1.0, 1.0)

def process_audio(audio_data: sr.AudioData) -> sr.AudioData:
    """
    Обрабатывает аудио:
    1. Денойз нейросетью (pyrnnoise)
    2. AGC – усиление тихого голоса
    """
    # Получаем сырые int16 данные (моно)
    raw = audio_data.get_raw_data()
    samples_in = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Для denoise_chunk нужен int16 массив формы [num_channels, num_samples]
    audio_int16 = (np.clip(samples_in, -1, 1) * 32767).astype(np.int16)

    # Денойз (pyrnnoise ожидает int16)
    # denoise_chunk обрабатывает весь массив и возвращает покадрово.
    # Мы склеим все выходные кадры в один массив.
    input_tensor = np.expand_dims(audio_int16, axis=0)  # (1, num_samples)
    output_frames = []
    for _, denoised_frame in denoiser.denoise_chunk(input_tensor):
        # denoised_frame: (1, frame_size) int16
        output_frames.append(denoised_frame.flatten())

    if output_frames:
        denoised_int16 = np.concatenate(output_frames)
    else:
        denoised_int16 = audio_int16  # fallback

    # Переводим в float32 для AGC
    denoised_float = denoised_int16.astype(np.float32) / 32768.0

    # AGC усиление
    enhanced_float = apply_agc(denoised_float)

    # Обратно в int16
    enhanced_int16 = (np.clip(enhanced_float, -1, 1) * 32767).astype(np.int16)
    return sr.AudioData(enhanced_int16.tobytes(), SAMPLE_RATE, 2)

def callback(recognizer, audio):
    """Фоновая обработка каждой порции аудио."""
    try:
        cleaned = process_audio(audio)
        text = recognizer.recognize_google(cleaned, language="ru-RU")
        print(f"User said: {text}")
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")

def main():
    r = sr.Recognizer()
    r.dynamic_energy_threshold = True
    r.energy_threshold = 4000
    r.pause_threshold = 1.2
    r.phrase_threshold = 0.2

    mic = sr.Microphone(device_index=1, sample_rate=SAMPLE_RATE)

    # Калибровка фона
    with mic as source:
        print("Калибровка шума...")
        r.adjust_for_ambient_noise(source, duration=1)
        # После очистки нейросетью шум почти исчезнет – можно смело снижать порог
        r.energy_threshold = max(r.energy_threshold * 0.25, 80)
        print(f"Порог энергии: {r.energy_threshold:.1f}")

    print("Слушаю... (нажмите Enter для выхода)")
    stop = r.listen_in_background(mic, callback)
    input()
    stop(wait_for_stop=True)
    print("Микрофон освобождён.")

if __name__ == "__main__":
    main()