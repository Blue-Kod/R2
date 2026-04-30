#!/usr/bin/env python3
"""
Умный захват речи: детектор тишины + Google STT.
Микрофон 48000 Гц, автоматическое определение конца фразы,
вывод задержки между концом речи и результатом распознавания.
"""

import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from scipy.signal import resample_poly

# ---------- настройки ----------
DEVICE = 1                # индекс микрофона
SAMPLE_RATE = 48000       # частота захвата
TARGET_RATE = 16000       # частота для Google STT

# VAD (Voice Activity Detection)
SILENCE_THRESH = 0.02     # порог RMS для тишины (подберите под ваш микрофон)
SILENCE_LIMIT = 1.0       # секунд тишины, после которых считаем фразу законченной
SPEECH_MIN_DUR  = 0.3     # минимальная длина фразы (игнорируем щелчки)

# общие
CHUNK_SEC = 0.1           # размер обрабатываемого кусочка (секунд)
# ---------------------------------

class ContinuousListener:
    def __init__(self, device, sample_rate, target_rate):
        self.device = device
        self.sample_rate = sample_rate
        self.target_rate = target_rate
        self.recognizer = sr.Recognizer()
        self.stream = None

    def start(self):
        """Открывает микрофон."""
        self.stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            dtype='int16',
            blocksize=int(self.sample_rate * CHUNK_SEC)
        )
        self.stream.start()

    def stop(self):
        """Корректно закрывает микрофон."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def rms(self, data):
        """Среднеквадратичная энергия для int16."""
        return np.sqrt(np.mean(data.astype(np.float32) ** 2))

    def resample_to_target(self, audio):
        """Полифазный ресемплинг int16 -> int16."""
        # resample_poly работает с float, поэтому конвертируем туда-обратно
        audio_float = audio.astype(np.float32) / 32768.0
        resampled = resample_poly(audio_float, self.target_rate, self.sample_rate)
        # обратно в int16
        resampled_int = (resampled * 32767).astype(np.int16)
        return np.clip(resampled_int, -32768, 32767)

    def listen_for_phrase(self):
        """
        Слушает до тех пор, пока не начнётся речь, а затем не наступит пауза.
        Возвращает аудио-фрагмент (int16, target_rate) или None.
        """
        audio_buffer = []
        silence_frames = 0
        speech_frames = 0
        speaking = False
        frames_per_silence = int(SILENCE_LIMIT / CHUNK_SEC)

        print("Ожидаю речь...", end="", flush=True)
        while True:
            chunk, _ = self.stream.read(int(self.sample_rate * CHUNK_SEC))
            chunk = chunk.flatten()
            energy = self.rms(chunk)

            if energy > SILENCE_THRESH:
                # Голос
                if not speaking:
                    print("\nГоворите...", end="", flush=True)
                    speaking = True
                    audio_buffer = []  # очищаем предыдущую тишину
                silence_frames = 0
                speech_frames += 1
                audio_buffer.append(chunk)
            else:
                # Тишина
                if speaking:
                    silence_frames += 1
                    audio_buffer.append(chunk)  # сохраняем хвост тишины (помогает распознаванию)
                    if silence_frames >= frames_per_silence:
                        # Проверяем минимальную длительность речи
                        if speech_frames * CHUNK_SEC >= SPEECH_MIN_DUR:
                            break
                        else:
                            # Слишком коротко – игнорируем, ждём заново
                            print("\n(короткий звук, игнорирую) Ожидаю речь...", end="", flush=True)
                            speaking = False
                            audio_buffer = []
                            silence_frames = 0
                            speech_frames = 0
                else:
                    # Тишина вне речи – просто пропускаем
                    pass

        # Конец фразы
        phrase_end_time = time.time()
        print(" Обработка...", end="", flush=True)

        # Объединяем буфер и ресемплируем в 16 кГц
        raw_audio = np.concatenate(audio_buffer)
        audio_16k = self.resample_to_target(raw_audio)
        return audio_16k, phrase_end_time

    def recognize_google(self, audio_16k):
        """Отправляет аудио в Google STT и возвращает текст + время получения."""
        start_recog = time.time()
        audio_data = sr.AudioData(audio_16k.tobytes(), self.target_rate, 2)
        try:
            text = self.recognizer.recognize_google(audio_data, language="ru-RU")
            recog_time = time.time()
            return text, recog_time
        except sr.UnknownValueError:
            return None, time.time()
        except sr.RequestError as e:
            print(f"\nОшибка Google: {e}")
            return None, time.time()

    def run_forever(self):
        """Вечный цикл распознавания с выводом задержки."""
        self.start()
        print("Микрофон активирован. Для выхода нажмите Ctrl+C.\n")
        try:
            while True:
                phrase_audio, phrase_end = self.listen_for_phrase()
                if phrase_audio is None:
                    continue
                text, recog_end = self.recognize_google(phrase_audio)
                delay = recog_end - phrase_end
                if text:
                    print(f"\n✅ Распознано: \"{text}\"")
                else:
                    print("\n🤷 Не распознано")
                print(f"⏱️ Задержка от конца речи до ответа: {delay:.2f} сек\n")
        except KeyboardInterrupt:
            print("\nЗавершение...")
        finally:
            self.stop()
            print("Микрофон освобождён.")

# ------------------ запуск ------------------
if __name__ == "__main__":
    listener = ContinuousListener(DEVICE, SAMPLE_RATE, TARGET_RATE)
    listener.run_forever()