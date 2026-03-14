import time
import wave
import threading
import queue
import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig


class TTS:
    """Стриминг TTS с использованием sounddevice"""

    def __init__(self, model_path):
        self.voice = PiperVoice.load(model_path)
        self.syn_config = SynthesisConfig(
            volume=1,  # half as loud
            length_scale=1,  # twice as slow
            noise_scale=1.0,  # more audio variation
            noise_w_scale=1.0,  # more speaking variation
            normalize_audio=False,
        )

        self.audio_queue = queue.Queue()
        self.is_playing = True
        self.stream = None
        self.sample_rate = None
        self.channels = None
        self.dtype = None

    def audio_player(self):
        """Функция для воспроизведения из очереди"""
        # Ждём первый чанк, чтобы узнать параметры
        first_chunk = self.audio_queue.get()

        self.sample_rate = first_chunk.sample_rate
        self.channels = first_chunk.sample_channels
        # Piper отдаёт int16, в sounddevice это соответствует 'int16'
        self.dtype = 'int16'

        # Создаём выходной поток
        with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=2048  # размер блока для низкой задержки
        ) as stream:
            self.stream = stream

            # Воспроизводим первый чанк
            stream.write(np.frombuffer(first_chunk.audio_int16_bytes, dtype=self.dtype))

            # Продолжаем воспроизводить остальные чанки
            while self.is_playing or not self.audio_queue.empty():
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    audio_array = np.frombuffer(chunk.audio_int16_bytes, dtype=self.dtype)
                    stream.write(audio_array)
                except queue.Empty:
                    continue

    def synthesize_and_play(self, text):
        """Синтезирует и сразу воспроизводит текст"""
        player_thread = threading.Thread(target=self.audio_player)
        player_thread.start()

        t1 = time.time()
        chunks_count = 0

        for chunk in self.voice.synthesize(text, self.syn_config):
            self.audio_queue.put(chunk)
            chunks_count += 1
        self.is_playing = False
        player_thread.join()



if __name__ == "__main__":
    model_path = "voices/irina-ru.onnx"
    text = "Здравствуйте! Чем могу помочь?"
    player = TTS(model_path)
    print("TTS начат")
    player.synthesize_and_play(text)
    print("TTS заврешён")
