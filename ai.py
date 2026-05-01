"""
AI Library for R2 Robot – persistent Gemini Live session.
Supports continuous video feed and text commands.
"""

import asyncio
import json
import base64
import re
import traceback
import time
import threading
from typing import Optional

import cv2
import numpy as np
import sounddevice as sd
import websockets

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
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
WS_URL = "wss://proxy-gemini-rlj1.onrender.com/ws/live"
VOICE = "Enceladus"
MAX_HISTORY_CHARS = 8000

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_websocket = None                     # текущий WebSocket
_stop_event = asyncio.Event()         # сигнал завершения
_audio_output_stream = None
_audio_enabled = True
_chat_history = []

# Для ожидания ответов на текстовые команды
_response_events = {}                 # id -> threading.Event
_response_texts = {}                  # id -> str
_response_counter = 0
_lock = threading.Lock()

# Очередь сообщений, которые нужно отправить в WebSocket (из любого потока)
_send_queue: Optional[asyncio.Queue] = None

# -------------------------------------------------------------------
# High‑level функции для AI (без изменений)
# -------------------------------------------------------------------
def log(message: str) -> None:
    _chat_history.append(f"[LOG]: {message}")
    print(f"[LOG]: {message}")

def set_emote(emotion: str) -> bool:
    from r2_app.high_level import emote
    return emote(emotion)

def set_eyes(x: float, y: float) -> None:
    from r2_app.high_level import set_eyes_position
    set_eyes_position(x, y)

def get_cpu_temp() -> str:
    from r2_app.high_level import cpu_temp
    return cpu_temp()

def get_system_stats() -> dict:
    from r2_app.high_level import health_snapshot
    return health_snapshot()

_AI_EXEC_GLOBALS = {
    "log": log, "print": log, "set_emote": set_emote,
    "set_eyes": set_eyes, "get_cpu_temp": get_cpu_temp,
    "get_system_stats": get_system_stats,
}

_execution_logs = []

async def execute_python(code: str) -> str:
    global _execution_logs
    _execution_logs = []
    print(f"\n--- [AI EXEC] ---\n{code}\n-----------------")
    try:
        exec(code, {"__builtins__": __builtins__}, _AI_EXEC_GLOBALS)
        return "\n".join(_execution_logs) if _execution_logs else "Код выполнен."
    except Exception:
        return f"ОШИБКА В КОДЕ:\n{traceback.format_exc()}"

# -------------------------------------------------------------------
# Системный промпт
# -------------------------------------------------------------------
CHARACTER_PROMPT = """
Ты — робот R2. Стиль: краткий, немного официальный.
Говори на русском. Всегда используй '+' перед ударными гласными. Мужской род.
Говор+и внятно, не тор+опся. Произнос+и слов+а полн+остью.

Ты видишь мир через камеру. Перед каждым ответом тебе показывают кадры.
Используй это, чтобы описывать окружение, людей, предметы.

ИНСТРУМЕНТЫ (Python):
- log(msg)
- set_emote(emotion)
- set_eyes(x, y)
- get_cpu_temp()
- get_system_stats()

АЛГОРИТМ:
1. Если нужен код: напиши #EXECUTE ... #END и замолчи.
2. После получения [SYSTEM_LOGS] ответь обычным текстом.
3. Ответ без #EXECUTE – конец задачи.
"""

CONFIG_BASE = {
    "response_modalities": ["AUDIO"],
    "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": VOICE}}},
    "safety_settings": [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ],
    "system_instruction": CHARACTER_PROMPT
}

def build_config():
    history_text = "\n".join(_chat_history)[-MAX_HISTORY_CHARS:] if _chat_history else ""
    prompt = CHARACTER_PROMPT + ("\n\nКОНТЕКСТ:\n" + history_text if history_text else "")
    cfg = dict(CONFIG_BASE)
    cfg["system_instruction"] = prompt
    return cfg

# -------------------------------------------------------------------
# Вспомогательные функции обработки ответов
# -------------------------------------------------------------------
def _play_audio(audio_b64: str):
    """Декодирует и воспроизводит аудио (вызывается из event loop)."""
    if not _audio_enabled or _audio_output_stream is None:
        return
    try:
        missing = len(audio_b64) % 4
        if missing:
            audio_b64 += '=' * (4 - missing)
        raw = base64.b64decode(audio_b64)
        if len(raw) % 2:
            raw = raw[:-1]
        arr = np.frombuffer(raw, dtype=np.int16)
        if arr.size > 0:
            upsampled = np.repeat(arr, 2)
            _audio_output_stream.write(upsampled)
    except Exception as e:
        print(f"[Audio] Ошибка воспроизведения: {e}")

# -------------------------------------------------------------------
# Асинхронные задачи внутри event loop
# -------------------------------------------------------------------
async def _process_incoming(websocket):
    """Читает сообщения от прокси, раздаёт текст/аудио."""
    global _response_texts, _response_events, _chat_history, _current_response

    async for raw in websocket:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        text_chunk = msg.get("text")
        audio_b64 = msg.get("audio")
        end_of_turn = msg.get("end_of_turn", False)

        # Сборка полного текста ответа
        if text_chunk:
            clean = re.sub(r'\+(?=[а-яёa-z])', '', text_chunk, flags=re.IGNORECASE)
            # Ищем активный запрос (последний добавленный)
            with _lock:
                keys = list(_response_texts.keys())
                if keys:
                    last_id = keys[-1]
                    _response_texts[last_id] = _response_texts.get(last_id, "") + clean

        if audio_b64:
            await asyncio.to_thread(_play_audio, audio_b64)

        if end_of_turn:
            # Завершился ответ – разблокируем ожидающий поток (для последнего запроса)
            with _lock:
                keys = list(_response_events.keys())
                if keys:
                    last_id = keys[-1]
                    ev = _response_events.pop(last_id, None)
                    if ev:
                        ev.set()

async def _process_outgoing(websocket, queue: asyncio.Queue):
    """Отправляет всё из очереди в WebSocket."""
    while True:
        msg = await queue.get()
        if msg is None:          # сигнал завершения
            break
        await websocket.send(json.dumps(msg))

async def _session_main():
    """Главная корутина: соединяется, держит сессию, слушает очередь."""
    global _websocket, _send_queue, _stop_event

    # Инициализация аудиовыхода (если включён)
    global _audio_output_stream
    if _audio_output_stream is None and _audio_enabled:
        _audio_output_stream = sd.OutputStream(
            samplerate=HARDWARE_RATE, channels=1, dtype='int16',
            device=1, latency='low')
        _audio_output_stream.start()

    _send_queue = asyncio.Queue()

    try:
        async with websockets.connect(WS_URL, open_timeout=30) as ws:
            _websocket = ws
            # Отправляем конфиг
            await ws.send(json.dumps({
                "api_key": API_KEY,
                "model_id": MODEL_ID,
                "config": build_config()
            }))
            print("[AI] Сессия Gemini Live запущена.")

            # Запускаем обработчики
            incoming = asyncio.create_task(_process_incoming(ws))
            outgoing = asyncio.create_task(_process_outgoing(ws, _send_queue))

            # Ожидаем сигнала остановки
            await _stop_event.wait()

            # Корректное завершение
            await _send_queue.put(None)   # сигнал outgoing‑таске
            incoming.cancel()
            outgoing.cancel()
            try:
                await asyncio.gather(incoming, outgoing, return_exceptions=True)
            except Exception:
                pass

    except Exception as e:
        print(f"[AI] Критическая ошибка сессии: {e}")
    finally:
        _websocket = None
        print("[AI] Сессия завершена.")

def _run_event_loop():
    """Точка входа для фонового потока."""
    global _loop, _stop_event
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _stop_event = asyncio.Event()
    _loop.run_until_complete(_session_main())

# -------------------------------------------------------------------
# Публичный интерфейс (вызывается из main.py)
# -------------------------------------------------------------------
def start_ai():
    """Запускает фоновую сессию. Можно вызывать один раз при старте."""
    global _loop_thread
    if _loop_thread and _loop_thread.is_alive():
        return
    print("[AI] Запуск фоновой сессии...")
    _loop_thread = threading.Thread(target=_run_event_loop, daemon=True)
    _loop_thread.start()
    # Даём немного времени на установку соединения
    time.sleep(1.0)

def send_frame():
    """
    Отправляет текущий кадр с основной камеры.
    Безопасно вызывать из любого потока, в том числе каждую секунду.
    """
    if _loop is None or _send_queue is None:
        return
    try:
        from r2_app.high_level import get_raw_frame
        frame = get_raw_frame(left=True)
        if frame is None:
            return
        ret, jpeg = cv2.imencode('.jpg', frame,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            return
        b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        # Отправляем в event loop
        asyncio.run_coroutine_threadsafe(
            _send_queue.put({"image_b64": b64, "mime_type": "image/jpeg"}),
            _loop
        )
    except Exception as e:
        print(f"[send_frame] Ошибка: {e}")

def command(text: str) -> str:
    """
    Отправляет текст в ИИ и БЛОКИРУЕТСЯ до получения полного ответа.
    Возвращает ответ (текст без маркеров ударения).
    """
    if _loop is None or _send_queue is None:
        return "AI не запущен. Вызовите start_ai()."

    # Регистрируем будущий ответ
    global _response_counter
    with _lock:
        msg_id = _response_counter
        _response_counter += 1
        ev = threading.Event()
        _response_events[msg_id] = ev
        _response_texts[msg_id] = ""

    # Кладём сообщение в очередь
    asyncio.run_coroutine_threadsafe(
        _send_queue.put({"text": text}),
        _loop
    )

    # Ожидаем ответа (таймаут 60 секунд)
    if ev.wait(timeout=60.0):
        with _lock:
            resp = _response_texts.pop(msg_id, "")
        return resp if resp else "Нет текстового ответа."
    else:
        with _lock:
            _response_events.pop(msg_id, None)
            _response_texts.pop(msg_id, None)
        return "Таймаут ожидания ответа."

def enable_audio(enabled: bool):
    global _audio_enabled
    _audio_enabled = enabled

def cleanup():
    global _stop_event, _loop
    print("[AI] Завершение сессии...")
    if _loop is not None:
        # Устанавливаем событие остановки
        asyncio.run_coroutine_threadsafe(_set_stop(), _loop)
    if _loop_thread and _loop_thread.is_alive():
        _loop_thread.join(timeout=5.0)
    if _audio_output_stream:
        _audio_output_stream.stop()
        _audio_output_stream.close()

async def _set_stop():
    global _stop_event
    _stop_event.set()

# Для обратной совместимости с main.py: он вызывает command() и потом что-то ещё.
# Если хотите полностью новое поведение, main.py надо чуть поправить (см. ниже).