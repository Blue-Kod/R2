#!/usr/bin/env python3
import sounddevice as sd
import speech_recognition as sr
import numpy as np
import time

# Настройки
DEVICE = 1
SAMPLE_RATE = 48000
TARGET_RATE = 16000
SILENCE_THRESHOLD = 0.02       # RMS порог тишины (подберите под микрофон, если нужно)
SILENCE_DURATION = 1.0         # секунд тишины для завершения фразы
CHUNK_DURATION = 0.1           # длительность маленького чанка для анализа

class RealtimePhraseRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.silence_threshold = SILENCE_THRESHOLD
        self.silence_duration = SILENCE_DURATION
        self.chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)

        self.phrase_buffer = np.array([], dtype=np.int16)
        self.is_speaking = False
        self.silence_start = None
        self.last_phrase_end_time = None

    def process_chunk(self, chunk):
        """Принимает numpy-чанк int16, обновляет состояние VAD."""
        rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))

        if rms > self.silence_threshold:
            # Речь
            if not self.is_speaking:
                print("🎤 Фраза началась...")
                self.is_speaking = True
                self.phrase_buffer = chunk
            else:
                self.phrase_buffer = np.concatenate([self.phrase_buffer, chunk])
            self.silence_start = None
        else:
            # Тишина
            if self.is_speaking:
                if self.silence_start is None:
                    self.silence_start = time.time()
                elif time.time() - self.silence_start >= self.silence_duration:
                    # Конец фразы
                    phrase_end_time = time.time()
                    print("⏹ Конец фразы, распознаю...")
                    text, latency = self.recognize_phrase(self.phrase_buffer, phrase_end_time)
                    if text:
                        print(f"✅ Распознано: '{text}'")
                        print(f"⏱ Задержка от конца речи до результата: {latency:.3f} сек")
                    else:
                        print("🤷 Не удалось разобрать")
                    # Сброс
                    self.is_speaking = False
                    self.phrase_buffer = np.array([], dtype=np.int16)
                    self.silence_start = None
                    return True  # фраза завершена
        return False

    def recognize_phrase(self, audio_int16, phrase_end_time):
        """Ресемплирует, отправляет в Google, возвращает (text, latency)."""
        # Ресемплинг 48k -> 16k (простой дециматор)
        audio_16k = audio_int16[::SAMPLE_RATE // TARGET_RATE]

        # Создаём AudioData
        audio_data = sr.AudioData(audio_16k.tobytes(), TARGET_RATE, 2)

        start_recognition = time.time()
        try:
            text = self.recognizer.recognize_google(audio_data, language="ru-RU")
            latency = time.time() - phrase_end_time
            return text, latency
        except sr.UnknownValueError:
            return None, 0
        except sr.RequestError as e:
            print(f"🌐 Ошибка сети: {e}")
            return None, 0
        except Exception as e:
            print(f"Ошибка распознавания: {e}")
            return None, 0

def main():
    print("Запуск непрерывного распознавания...")
    print(f"Микрофон: устройство {DEVICE}, порог тишины {SILENCE_THRESHOLD}, ожидание тишины {SILENCE_DURATION} с")
    print("Нажмите Ctrl+C для выхода.\n")

    phrase_recognizer = RealtimePhraseRecognizer()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"Ошибка потока: {status}")
        # indata shape: (frames, 1), float32 -> int16
        chunk = (indata[:, 0] * 32767).astype(np.int16)
        phrase_recognizer.process_chunk(chunk)

    stream = sd.InputStream(
        device=DEVICE,
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype='float32',
        callback=audio_callback,
        blocksize=phrase_recognizer.chunk_samples
    )
    stream.start()

    try:
        while True:
            time.sleep(0.1)  # освобождаем главный поток
    except KeyboardInterrupt:
        print("\nОстановка...")
    finally:
        stream.stop()
        stream.close()
        print("Микрофон освобождён.")

if __name__ == "__main__":
    main()