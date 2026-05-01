"""
AI Library for R2 Robot – Persistent, self-healing session with Gemini Live.
- command(text)  -> отправить текст модели (асинхронно, ответ обрабатывается полностью)
- send_frame()   -> отправить кадр с камеры
- Сессия автоматически переподключается при обрыве.
- Модель может выполнять Python-код через #EXECUTE блоки.
"""

import asyncio
import json
import base64
import re
import traceback
import threading
import time
import requests
import numpy as np
import sounddevice as sd
import websockets
import cv2

# --- Configuration ---
OBSCURED_API_KEY = "c1RZQWNjZG8xT3ZHMV9IdldFVTMzakNfU3dhQ19PVWtEeVNheklB"

def get_real_key(obscured):
    try:
        decoded = base64.b64decode(obscured).decode()
        return decoded[::-1]
    except Exception:
        return None

API_KEY = get_real_key(OBSCURED_API_KEY)
MODEL_ID = "gemini-3.1-flash-live-preview"
MODEL_RATE = 24000
HARDWARE_RATE = 48000
BASE_URL = "https://proxy-gemini-rlj1.onrender.com"
WS_URL = "wss://proxy-gemini-rlj1.onrender.com/ws/live"
VOICE = "Enceladus"
MAX_HISTORY_CHARS = 8000

# Global state
_audio_enabled = True
_chat_history = []
_output_stream = None
_current_response = ""

# --- High-level functions available to AI ---
def log(message: str) -> None:
    formatted_msg = f"[LOG]: {message}"
    _chat_history.append(formatted_msg)
    print(formatted_msg)

def set_emote(emotion: str) -> bool:
    from r2_app.high_level import emote
    print(f"[AI-EXEC] Setting emote: {emotion}")
    return emote(emotion)

def set_eyes(x: float, y: float) -> None:
    from r2_app.high_level import set_eyes_position
    print(f"[AI-EXEC] Setting eyes: x={x}, y={y}")
    set_eyes_position(x, y)

def get_cpu_temp() -> str:
    from r2_app.high_level import cpu_temp
    temp = cpu_temp()
    log(temp)
    return temp

def get_system_stats() -> dict:
    from r2_app.high_level import health_snapshot
    stats = health_snapshot()
    log(stats)
    return stats

_AI_EXEC_GLOBALS = {
    "log": log,
    "print": log,
    "set_emote": set_emote,
    "set_eyes": set_eyes,
    "get_cpu_temp": get_cpu_temp,
    "get_system_stats": get_system_stats,
}

_execution_logs = []

def _exec_log(message):
    formatted = f"[LOG]: {message}"
    _execution_logs.append(formatted)
    print(formatted)

# --- Character Prompt ---
CHARACTER_PROMPT = """
Ты — робот R2. Стиль: краткий, немного официальный.
Говори на русском. Всегда используй '+' перед ударными гласными. Мужской род.
Говор+и внятно, не тор+опся. Произнос+и слов+а полн+остью, избег+ай сокращ+ений. Твоя речь должна быть разборчивой, как у диктора.

Ты видишь мир через камеру робота. Кадры приходят регулярно, используй их, чтобы описывать окружение, людей, предметы. Если видишь человека, поздоровайся или спроси, чем помочь.

ИНСТРУМЕНТЫ (Python):
- log(msg) -> запись в лог. Пользователь её НЕ видит.
- set_emote(emotion) -> установить эмоцию (happy, sad, neutral, и т.д.).
- set_eyes(x, y) -> положение глаз (от -1.0 до 1.0).
- get_cpu_temp() -> температура CPU.
- get_system_stats() -> статистика системы.

АЛГОРИТМ РЕКУРСИИ:
1. Если нужно действие/расчет: скажи "Выполн+яю..." и напиши #EXECUTE ... #END.
2. В блоке #EXECUTE пиши ТОЛЬКО код.
3. Если ты получила [SYSTEM_LOGS], проанализируй их и дай ответ пользователю ОБЫЧНЫМ ТЕКСТОМ.
4. ТВОЙ ОТВЕТ БЕЗ БЛОКА #EXECUTE ЯВЛЯЕТСЯ СИГНАЛОМ ЗАВЕРШЕНИЯ ЗАДАЧИ.
5. Пиши код только если тебе реально нужно получить данные или выявить команду.
6. Не пиши ничего после блока кода. Закончился блок кода - ВСЁ. КОНЕЦ. Только после следующего сообщения можешь что-то писать.
"""

CONFIG = {
    "response_modalities": ["AUDIO"],
    "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": VOICE}}},
    "system_instruction": CHARACTER_PROMPT
}

BANNED_PHRASES = ["i'm"]

async def execute_python(code):
    """Выполнить Python-код от имени робота. Возвращает строку-отчёт."""
    global _execution_logs
    _execution_logs = []
    print(f"\n--- [AI EXECUTION START] ---\n{code}\n-------------------------")
    try:
        exec(code, {"__builtins__": __builtins__}, _AI_EXEC_GLOBALS)
        return "\n".join(_execution_logs) if _execution_logs else "Код выполнен."
    except Exception:
        err = traceback.format_exc()
        print(f"[AI ERROR]:\n{err}")
        return f"ОШИБКА В КОДЕ:\n{err}"


# --- AISession: постоянное соединение, диалоги и видео ---
class AISession:
    def __init__(self):
        self.ws = None
        self.send_queue = asyncio.Queue()        # сообщения для отправки (текст, видео, логи)
        self.incoming_queue = asyncio.Queue()    # сообщения от модели
        self.command_queue = asyncio.Queue()     # задания на диалог
        self.loop = None
        self.thread = None
        self.running = False
        self._output_stream = None

    async def _ensure_output_stream(self):
        if self._output_stream is None and _audio_enabled:
            self._output_stream = sd.OutputStream(
                samplerate=HARDWARE_RATE,
                channels=1,
                dtype='int16',
                device=1,
                latency='low'
            )
            self._output_stream.start()

    async def _connect(self):
        """Подключиться к прокси и отправить конфиг."""
        for attempt in range(1, 6):
            try:
                resp = requests.get(BASE_URL, timeout=10)
                if resp.status_code < 500:
                    break
            except requests.RequestException:
                pass
            if attempt < 5:
                await asyncio.sleep(2)
        ws = await websockets.connect(WS_URL, open_timeout=30)
        await ws.send(json.dumps({
            "api_key": API_KEY,
            "model_id": MODEL_ID,
            "config": CONFIG
        }))
        return ws

    async def _session_loop(self):
        """Главный цикл: поддерживать WebSocket и переподключаться."""
        while self.running:
            try:
                self.ws = await self._connect()
                print("[AI] Соединение с прокси установлено")

                # Запускаем воркеры
                receiver_task = asyncio.create_task(self._receiver())
                sender_task = asyncio.create_task(self._sender())
                dialog_manager = asyncio.create_task(self._dialog_manager())

                done, pending = await asyncio.wait(
                    [receiver_task, sender_task, dialog_manager],
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    exc = task.exception()
                    if exc:
                        print(f"[AI] Ошибка в задаче: {exc}")

            except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                print(f"[AI] Соединение потеряно: {e}")
            except Exception as e:
                print(f"[AI] Неожиданная ошибка: {e}")

            # Очищаем очереди, чтобы не было мусора после обрыва
            while not self.incoming_queue.empty():
                self.incoming_queue.get_nowait()
            while not self.command_queue.empty():
                self.command_queue.get_nowait()

            if self.running:
                wait = 2
                print(f"[AI] Повторное подключение через {wait} сек...")
                await asyncio.sleep(wait)

    async def _receiver(self):
        """Читает сообщения от WebSocket и кладёт во входящую очередь."""
        while True:
            try:
                raw = await self.ws.recv()
                data = json.loads(raw)
                await self.incoming_queue.put(data)
            except websockets.ConnectionClosed:
                print("[AI] WebSocket закрыт")
                break
            except Exception as e:
                print(f"[AI] Ошибка приёма: {e}")
                continue

    async def _sender(self):
        """Читает очередь отправки и шлёт в WebSocket."""
        while True:
            message = await self.send_queue.get()
            try:
                await self.ws.send(json.dumps(message))
            except Exception as e:
                print(f"[AI] Ошибка отправки: {e}")
                # Возвращаем сообщение обратно, чтобы отправить после переподключения
                await self.send_queue.put(message)
                break

    async def _dialog_manager(self):
        """Последовательно обрабатывает запросы из command_queue."""
        while True:
            text = await self.command_queue.get()
            # Добавляем в историю
            _chat_history.append(f"Пользователь: {text}")
            print(f"INPUT -> AI: {text}")

            # Запускаем диалог
            await self._handle_dialog(text)

    async def _handle_dialog(self, initial_text: str):
        """Обрабатывает один диалог: отправляет текст, получает ответы, выполняет код, до финала."""
        # Отправляем начальный текст
        await self.send_queue.put({"text": initial_text})

        full_text = ""
        while True:
            # Собираем ответные сообщения до turn_complete
            turn_text = ""
            while True:
                data = await self.incoming_queue.get()

                # Обрабатываем текстовый фрагмент
                if "text" in data:
                    chunk = data["text"]
                    turn_text += chunk
                    # Выводим в консоль без маркеров ударений
                    display_chunk = re.sub(r"\+", "", chunk)
                    print(f"AI: {display_chunk}", end="", flush=True)

                # Проигрываем аудио, если есть
                if "audio" in data and _audio_enabled:
                    await self._play_audio(data["audio"])

                # Если конец реплики, заканчиваем сбор
                if data.get("end_of_turn"):
                    print("")  # перевод строки
                    break

            # Обработка #EXECUTE
            code_match = re.search(r"#EXECUTE\n(.*?)\n#END", turn_text, re.DOTALL)
            if code_match:
                # Выполняем код и отправляем результат
                report = await execute_python(code_match.group(1).strip())
                _chat_history.append(f"[SYSTEM_LOGS]: {report}")
                await self.send_queue.put({"text": f"[SYSTEM_LOGS]:\n{report}"})
                # Продолжаем слушать ответ модели
                continue
            else:
                # Финальный ответ (без кода). Завершаем диалог.
                cleaned = re.sub(r"\+", "", turn_text.strip())
                _chat_history.append(f"Ассистент: {cleaned}")
                return

    async def _play_audio(self, audio_b64: str):
        """Декодирует и проигрывает аудио из base64."""
        try:
            missing_padding = len(audio_b64) % 4
            if missing_padding:
                audio_b64 += '=' * (4 - missing_padding)
            raw_bytes = base64.b64decode(audio_b64)
            if len(raw_bytes) % 2 != 0:
                raw_bytes = raw_bytes[:-1]
            audio_array = np.frombuffer(raw_bytes, dtype=np.int16)
            if audio_array.size > 0:
                await self._ensure_output_stream()
                if self._output_stream:
                    # Ресэмплинг 24k -> 48k (повтор сэмплов)
                    mono_48k = np.repeat(audio_array, 2)
                    self._output_stream.write(mono_48k)
        except Exception as e:
            print(f"\n[Ошибка аудио]: {e}")

    def start(self):
        """Запустить фоновый поток с event loop."""
        if self.running:
            return
        self.running = True

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            loop.run_until_complete(self._session_loop())
            self.running = False

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def send_command(self, text: str):
        """Потокобезопасно добавить команду в очередь диалогов."""
        if self.loop and self.running:
            asyncio.run_coroutine_threadsafe(self.command_queue.put(text), self.loop)

    def send_message(self, message: dict):
        """Потокобезопасно добавить сообщение в очередь отправки (видео и т.п.)."""
        if self.loop and self.running:
            asyncio.run_coroutine_threadsafe(self.send_queue.put(message), self.loop)

    def stop(self):
        self.running = False
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
        if self._output_stream:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None


# --- Глобальная сессия и публичные функции ---
_session = None
_lock = threading.Lock()

def _get_session():
    global _session
    with _lock:
        if _session is None or not _session.running:
            _session = AISession()
            _session.start()
        return _session

def command(text: str) -> None:
    """Отправить текст модели. Диалог обрабатывается асинхронно."""
    session = _get_session()
    session.send_command(text)

def send_frame() -> None:
    """Отправить текущий кадр с основной камеры."""
    from r2_app.high_level import get_raw_frame

    session = _get_session()
    frame = get_raw_frame(left=True)
    if frame is None:
        print("[send_frame] Кадр не получен")
        return

    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ret:
        print("[send_frame] Ошибка JPEG кодирования")
        return

    b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    session.send_message({
        "image_b64": b64,
        "mime_type": "image/jpeg"
    })

def cleanup():
    global _session
    if _session:
        _session.stop()
        _session = None
    global _output_stream
    if _output_stream:
        _output_stream.stop()
        _output_stream.close()
        _output_stream = None