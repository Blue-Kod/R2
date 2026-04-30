#!/usr/bin/env python3
import sys
import os
import io
import time

# 🔇 Подавляем ALSA-логи при импорте PyAudio (используется библиотекой speech_recognition)
_stderr_backup = sys.stderr
sys.stderr = io.StringIO()
import speech_recognition as sr
sys.stderr = _stderr_backup


def callback(recognizer, audio):
    """Вызывается из фонового потока при получении аудио."""
    try:
        # Распознаём русскую речь с помощью Google Speech Recognition
        text = recognizer.recognize_google(audio, language="ru-RU")
        print(f"User said: {text}")
    except sr.UnknownValueError:
        print("Не удалось распознать речь (возможно, слишком шумно или ничего не сказано)")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")


def main():
    recognizer = sr.Recognizer()

    # Автоматическое шумоподавление: настройка порога под текущий фоновый шум
    recognizer.dynamic_energy_threshold = True       # включено по умолчанию (но явно укажем)
    recognizer.energy_threshold = 300                # начальное значение, будет скорректировано
    recognizer.pause_threshold = 0.8                 # секунд тишины перед концом фразы

    # Микрофон: устройство с индексом 1, частота дискретизации 48000 Гц
    mic = sr.Microphone(device_index=1, sample_rate=48000)

    with mic as source:
        # Калибровка под фон (вентилятор и т.д.) — длится 1 секунду
        print("Калибровка шума... (говорить не нужно)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"Порог энергии установлен: {recognizer.energy_threshold:.1f}")

    print("Слушаю... (нажмите Enter для выхода)")

    # Запуск фонового прослушивания
    stop_listening = recognizer.listen_in_background(mic, callback)

    # Ожидание ввода пользователя
    input()

    # Корректная остановка и освобождение микрофона
    stop_listening(wait_for_stop=True)
    print("Микрофон освобождён. Выход.")


if __name__ == "__main__":
    main()