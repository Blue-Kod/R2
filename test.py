#!/usr/bin/env python3
import sys, os, time
import numpy as np
import rnnoise

# Полностью заглушаем ALSA‑логи
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)

import speech_recognition as sr

# ----- Настройки шумоподавления -----
PROP_DECREASE = 0.6          # 0.0..1.0 – сила подавления noisereduce (ниже → меньше агрессии)
USE_RNNOISE = True          # True, если хотите использовать rnnoise вместо noisereduce
# ------------------------------------


def reduce_noise(audio_data: sr.AudioData, sample_rate=48000):
    denoiser = rnnoise.RNNoise()
    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
    # RNNoise ожидает float32 в диапазоне -1..1
    audio_float = audio_np.astype(np.float32) / 32768.0
    # Обработка покадрово (размер кадра 480 семплов для 48 кГц)
    frame_size = 480
    output = np.zeros_like(audio_float)
    for i in range(0, len(audio_float) - frame_size + 1, frame_size):
        frame = audio_float[i:i+frame_size]
        output[i:i+frame_size] = denoiser.process_frame(frame)
    output_int16 = (np.clip(output, -1, 1) * 32767).astype(np.int16)
    return sr.AudioData(output_int16.tobytes(), sample_rate, 2)


def callback(recognizer, audio):
    try:
        cleaned = reduce_noise(audio, 48000)
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

    mic = sr.Microphone(device_index=1, sample_rate=48000)

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
        r.energy_threshold = max(r.energy_threshold * 0.3, 100)   # снижаем порог
        print(f"Порог энергии: {r.energy_threshold:.1f}")

    print("Слушаю... (Enter для выхода)")
    stop = r.listen_in_background(mic, callback)
    input()
    stop(wait_for_stop=True)
    print("Микрофон освобождён.")

if __name__ == "__main__":
    main()