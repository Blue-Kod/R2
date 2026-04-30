#!/usr/bin/env python3
import sys
import os
import io
import time
import numpy as np
import noisereduce as nr

# ---------- 1. Полное подавление ALSA-логов ----------
class FilterStderr:
    """Фильтрует stderr, убирая строки с 'ALSA'."""
    def __init__(self, stream):
        self.stream = stream
    def write(self, message):
        if "ALSA" not in message:
            self.stream.write(message)
    def flush(self):
        self.stream.flush()

sys.stderr = FilterStderr(sys.stderr)

# Теперь можно импортировать и работать спокойно
import speech_recognition as sr

# Глобальные переменные шумоподавления
noise_profile = None
noise_sample_duration = 2   # секунд записи образца шума


def reduce_noise(audio_data: sr.AudioData, sample_rate: int = 48000) -> sr.AudioData:
    """Спектральное шумоподавление с помощью noisereduce."""
    global noise_profile
    if noise_profile is None:
        return audio_data

    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

    reduced = nr.reduce_noise(
        y=audio_np,
        sr=sample_rate,
        y_noise=noise_profile,
        prop_decrease=0.9,
        n_fft=1024,
        win_length=1024,
        hop_length=512
    )

    reduced_int16 = (reduced * 32767).astype(np.int16)
    return sr.AudioData(reduced_int16.tobytes(), sample_rate, 2)


def capture_noise_profile(recognizer, mic):
    """Записывает образец шума (вентилятор) без речи."""
    global noise_profile
    print(f"Запись образца шума ({noise_sample_duration} сек, молчите)...")
    with mic as source:
        audio_sample = recognizer.listen(source, timeout=noise_sample_duration,
                                         phrase_time_limit=noise_sample_duration)
    audio_np = np.frombuffer(audio_sample.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
    noise_profile = audio_np
    print("Профиль шума сохранён.")


def callback(recognizer, audio):
    """Фоновая обработка полученного аудио."""
    try:
        cleaned = reduce_noise(audio, sample_rate=48000)
        text = recognizer.recognize_google(cleaned, language="ru-RU")
        print(f"User said: {text}")
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")


def main():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 4000
    recognizer.pause_threshold = 1.2
    recognizer.phrase_threshold = 0.2

    mic = sr.Microphone(device_index=1, sample_rate=48000)

    # Захват шума
    capture_noise_profile(recognizer, mic)

    # Калибровка
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"Порог энергии после калибровки: {recognizer.energy_threshold:.1f}")

    print("Слушаю... (нажмите Enter для выхода)")

    stop_listening = recognizer.listen_in_background(mic, callback)
    input()
    stop_listening(wait_for_stop=True)
    print("Микрофон освобождён. Выход.")


if __name__ == "__main__":
    main()