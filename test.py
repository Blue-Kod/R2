#!/usr/bin/env python3
import sys
import os
import io
import time
import numpy as np
import noisereduce as nr

# 🔇 Подавляем ALSA-логи при импорте PyAudio
_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
import speech_recognition as sr
sys.stderr = _stderr_backup

# ------------------------------------------------------------
# Глобальные переменные для хранения профиля шума
noise_profile = None        # массив частотного профиля шума
noise_sample_duration = 2   # сколько секунд записывать образец шума
# ------------------------------------------------------------

def reduce_noise(audio_data: sr.AudioData, sample_rate: int = 48000) -> sr.AudioData:
    """
    Применяет спектральное шумоподавление к аудио.
    Если есть noise_profile, использует его; иначе возвращает без изменений.
    """
    global noise_profile
    if noise_profile is None:
        return audio_data

    # Преобразуем AudioData в массив float32
    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0

    # Применяем noisereduce
    # prop_decrease - степень подавления (0.0 - нет, 1.0 - максимум)
    reduced = nr.reduce_noise(
        y=audio_np,
        sr=sample_rate,
        y_noise=noise_profile,        # предварительно записанный профиль шума
        prop_decrease=0.9,            # стараемся убрать шум на 90%
        n_fft=1024,                   # размер окна БПФ
        win_length=1024,
        hop_length=512
    )

    # Обратно в int16
    reduced_int16 = (reduced * 32767).astype(np.int16)
    return sr.AudioData(reduced_int16.tobytes(), sample_rate, 2)


def capture_noise_profile(recognizer, mic):
    """
    Записывает образец шума (только вентилятор, без речи) и сохраняет как профиль.
    """
    global noise_profile
    print(f"Запись образца шума ({noise_sample_duration} сек, молчите)...")
    with mic as source:
        # Отключаем калибровку на время записи образца, чтобы захватить чистый шум
        audio_sample = recognizer.listen(source, timeout=noise_sample_duration, phrase_time_limit=noise_sample_duration)
    # Конвертируем в numpy float32
    audio_np = np.frombuffer(audio_sample.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
    noise_profile = audio_np
    print("Профиль шума сохранён.")


def callback(recognizer, audio):
    """Вызывается из фонового потока при получении аудио."""
    try:
        # Применяем шумоподавление
        cleaned_audio = reduce_noise(audio, sample_rate=48000)

        # Для отладки можно сохранить файл (если нужно)
        # with open("debug_audio.wav", "wb") as f:
        #     f.write(cleaned_audio.get_wav_data())

        # Распознаём русскую речь
        text = recognizer.recognize_google(cleaned_audio, language="ru-RU")
        print(f"User said: {text}")
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")


def main():
    recognizer = sr.Recognizer()

    # Настройки, менее критичные, но оставим для подстраховки
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 4000
    recognizer.pause_threshold = 1.2
    recognizer.phrase_threshold = 0.2

    mic = sr.Microphone(device_index=1, sample_rate=48000)

    # --- Захват образца шума ---
    capture_noise_profile(recognizer, mic)

    # --- Калибровка окружающего шума (после записи профиля) ---
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