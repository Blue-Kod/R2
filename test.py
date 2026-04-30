#!/usr/bin/env python3
import time
import speech_recognition as sr

def callback(recognizer, audio):
    """Вызывается из фонового потока при получении аудио."""
    try:
        text = recognizer.recognize_google(audio, language="ru-RU")
        print(f"User said: {text}")
    except sr.UnknownValueError:
        print("Не удалось распознать речь")
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания: {e}")

def main():
    r = sr.Recognizer()
    # Микрофон с индексом 1, частота 48000 Гц
    mic = sr.Microphone(device_index=1, sample_rate=48000)

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)   # калибровка

    print("Слушаю... (нажмите Enter для выхода)")

    # Запуск фонового прослушивания
    stop_listening = r.listen_in_background(mic, callback)

    # Ждём ввода от пользователя
    input()

    # Останавливаем слушатель и дожидаемся полного освобождения микрофона
    stop_listening(wait_for_stop=True)
    print("Микрофон освобождён. Выход.")

if __name__ == "__main__":
    main()