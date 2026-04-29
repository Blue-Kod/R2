#!/usr/bin/env python3
"""
Скрипт для проверки распознавания речи с помощью sherpa-onnx и модели SenseVoice.
Просто выводит в консоль всё, что услышит микрофон.
"""

import sys
import sherpa_onnx
import sounddevice as sd
import numpy as np


def create_recognizer():
    # Путь к папке с распакованной моделью SenseVoice
    model_path = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

    # Создаем модель с поддержкой VAD (автоматически определяет конец фразы)
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path,
        sample_rate=16000,
        use_vad=True,
        language="auto",  # автоопределение языка (русский поддерживается)
    )
    return recognizer


def main():
    print("Загружаю модель SenseVoice...")
    recognizer = create_recognizer()

    # Ищем микрофон
    devices = sd.query_devices()
    input_device = None
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_device = i
            print(f"Буду использовать микрофон: [{i}] {dev['name']}")
            break
    if input_device is None:
        print("Не найден ни один микрофон!")
        sys.exit(1)

    sample_rate = 16000  # SenseVoice работает на 16kHz
    chunk_seconds = 0.5  # Размер чанка для отправки на распознавание
    samples_per_chunk = int(sample_rate * chunk_seconds)

    print("\n" + "=" * 50)
    print("НАЧАЛО ЗАПИСИ. Говорите в микрофон!")
    print("Для выхода нажмите Ctrl+C")
    print("=" * 50 + "\n")

    try:
        with sd.InputStream(
                device=input_device,
                channels=1,
                samplerate=sample_rate,
                dtype="float32"
        ) as stream:
            audio_buffer = np.array([], dtype=np.float32)

            while True:
                chunk, _ = stream.read(samples_per_chunk)
                chunk = chunk.flatten()
                audio_buffer = np.concatenate([audio_buffer, chunk])

                # Отправляем аудио на распознавание
                stream_obj = recognizer.create_stream()
                stream_obj.accept_waveform(sample_rate, audio_buffer)
                recognizer.decode_stream(stream_obj)

                text = stream_obj.result.text

                if text:
                    print(f"🎤 Распознано: {text}")
                    audio_buffer = np.array([], dtype=np.float32)  # очищаем буфер

    except KeyboardInterrupt:
        print("\nЗавершение записи...")


if __name__ == "__main__":
    main()