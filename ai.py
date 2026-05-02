"""
AI Library for R2 Robot - Persistent WebSocket session with reconnection,
text & frame history, thread-safe message queue.
"""

import asyncio
import base64
import json
import queue
import re
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import requests
import sounddevice as sd
import websockets

# ==============================================================================
# Configuration
# ==============================================================================
OBSCURED_API_KEY = "c1RZQWNjZG8xT3ZHMV9IdldFVTMzakNfU3dhQ19PVWtEeVNheklB"

def get_real_key(obscured):
    try:
        decoded = base64.b64decode(obscured).decode()
        return decoded[::-1]
    except Exception:
        return None

API_KEY = get_real_key(OBSCURED_API_KEY)
MODEL_ID = "gemini-3.1-flash-live-preview"
HARDWARE_RATE = 48000
BASE_URL = "https://proxy-gemini-rlj1.onrender.com"
WS_URL = "wss://proxy-gemini-rlj1.onrender.com/ws/live"
VOICE = "Enceladus"
MAX_HISTORY_CHARS = 8000
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BASE_DELAY = 2.0
FRAME_HISTORY_LENGTH = 15
KEEPALIVE_TIMEOUT_SECONDS = 300

# Global state
_audio_enabled = True
_chat_history = []
_output_stream = None
_current_response = ""
_session: Optional['AISession'] = None
_session_lock = threading.Lock()
_frame_buffer: deque[bytes] = deque(maxlen=FRAME_HISTORY_LENGTH)

def get_current_response():
    return _current_response

def enable_ai_audio(enabled: bool) -> None:
    global _audio_enabled
    _audio_enabled = bool(enabled)
    print(f"[AI] Audio {'enabled' if _audio_enabled else 'disabled'}")

# --- High-level functions available to AI (as before) ---
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
    "log": log, "print": log,
    "set_emote": set_emote, "set_eyes": set_eyes,
    "get_cpu_temp": get_cpu_temp, "get_system_stats": get_system_stats,
}

CHARACTER_PROMPT = """
Ты — робот R2. Стиль: краткий, немного официальный.
Говори на русском. Всегда используй '+' перед ударными гласными. Мужской род.
Говор+и внятно, не тор+опся. Произнос+и слов+а полн+остью, избег+ай сокращ+ений. Твоя речь должна быть разборчивой, как у диктора.

Ты видишь мир через камеру робота. Перед каждым ответом тебе показывают несколько кадров с основной камеры. Используй эту информацию, чтобы описывать окружение, людей, предметы. Если видишь человека, поздоровайся или спроси, чем помочь.

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

BANNED_PHRASES = ["i'm"]

def build_system_prompt(chat_history):
    if not chat_history: return CHARACTER_PROMPT
    history_text = "\n".join(chat_history)[-MAX_HISTORY_CHARS:]
    return f"{CHARACTER_PROMPT}\n\nКОНТЕКСТ ДИАЛОГА ДО ПОСЛЕДНЕГО СБОЯ:\n{history_text}"

def build_config(chat_history):
    config = dict(CONFIG_BASE)
    config["system_instruction"] = build_system_prompt(chat_history)
    return config

# ==============================================================================
# Enhanced AISession (Thread-safe message queue)
# ==============================================================================

class AISession:
    def __init__(self):
        self.ws = None
        self.send_queue = queue.Queue()
        self.loop = None
        self.thread = None
        self.running = False
        self._output_stream = None
        self._session_token: Optional[str] = None

    async def _ensure_output_stream(self):
        if self._output_stream is None and _audio_enabled:
            self._output_stream = sd.OutputStream(
                samplerate=HARDWARE_RATE, channels=1, dtype='int16', device=1, latency='low'
            )
            self._output_stream.start()

    async def _run_session(self):
        # Warm up proxy
        for attempt in range(1, 6):
            try:
                resp = requests.get(BASE_URL, timeout=10)
                if resp.status_code < 500:
                    print(f"Warmed up Render ({resp.status_code})")
                    break
            except requests.RequestException: pass
            if attempt < 5: await asyncio.sleep(2)

        reconnect_attempt = 0
        while self.running and reconnect_attempt < MAX_RECONNECT_ATTEMPTS:
            try:
                await self._connect_and_process()
                break
            except (websockets.ConnectionClosed, asyncio.TimeoutError, OSError) as e:
                reconnect_attempt += 1
                print(f"Connection lost (attempt {reconnect_attempt}/{MAX_RECONNECT_ATTEMPTS}): {e}")
                delay = RECONNECT_BASE_DELAY * reconnect_attempt
                time.sleep(delay)
            except Exception as e:
                print(f"Unexpected error in session: {e}")
                break
        self.running = False

    async def _connect_and_process(self):
        config = build_config(_chat_history)
        config["session_resumption_config"] = {
            "maximum_timeout": str(KEEPALIVE_TIMEOUT_SECONDS) + "s"
        }
        setup_msg = {"api_key": API_KEY, "model_id": MODEL_ID, "config": config}
        if self._session_token:
            setup_msg["session_resumption"] = {"handle": self._session_token}

        async with websockets.connect(WS_URL, open_timeout=30) as ws:
            self.ws = ws
            await ws.send(json.dumps(setup_msg))
            print("WebSocket session opened")

            if _chat_history:
                await self._send_text_history(ws)
            if _frame_buffer:
                await self._send_buffered_frames(ws)

            receiver_task = asyncio.create_task(self._receiver())
            sender_task = asyncio.create_task(self._sender())
            await asyncio.gather(receiver_task, sender_task)

    async def _send_text_history(self, ws):
        turns = []
        for msg in _chat_history:
            role = "user" if msg.startswith("Пользователь:") else "model"
            turns.append({"role": role, "parts": [{"text": msg}]})
        msg = {"clientContent": {"turns": turns, "turn_complete": True}}
        await ws.send(json.dumps(msg))
        print("Injected text history into session.")

    async def _send_buffered_frames(self, ws):
        print(f"Re-sending {len(_frame_buffer)} buffered frames...")
        for jpeg_bytes in _frame_buffer:
            b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            await ws.send(json.dumps({"image_b64": b64, "mime_type": "image/jpeg"}))
            await asyncio.sleep(0.5)
        print("Buffered frames sent.")

    async def _receiver(self):
        global _current_response
        while True:
            try:
                raw = await self.ws.recv()
                data = json.loads(raw)
            except websockets.ConnectionClosed as e:
                print(f"Receiver: Connection closed ({e})")
                break
            except Exception as e:
                print(f"Receiver error: {e}")
                break

            if "sessionResumption" in data:
                self._session_token = data["sessionResumption"]["handle"]

            if "text" in data:
                clean = re.sub(r"\+", "", data["text"])
                _current_response += clean
                print(clean, end="", flush=True)

            if "audio" in data and _audio_enabled:
                try:
                    audio_b64 = data["audio"]
                    missing = len(audio_b64) % 4
                    if missing: audio_b64 += '=' * (4 - missing)
                    raw_bytes = base64.b64decode(audio_b64)
                    if len(raw_bytes) % 2: raw_bytes = raw_bytes[:-1]
                    arr = np.frombuffer(raw_bytes, dtype=np.int16)
                    if arr.size > 0:
                        await self._ensure_output_stream()
                        if self._output_stream:
                            self._output_stream.write(np.repeat(arr, 2))
                except Exception as e:
                    print(f"\nAudio error: {e}")

            if data.get("end_of_turn"):
                print()
                _current_response = ""

    async def _sender(self):
        while True:
            try:
                msg = self.send_queue.get_nowait()
                try:
                    await self.ws.send(json.dumps(msg))
                except Exception as e:
                    print(f"Send error: {e}")
            except queue.Empty:
                pass
            await asyncio.sleep(0.05)

    def start(self):
        if self.running: return
        self.running = True
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            try:
                loop.run_until_complete(self._run_session())
            finally:
                loop.close()
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

    def send_message(self, msg):
        if self.running:
            self.send_queue.put(msg)

    def stop(self):
        self.running = False
        if self.loop and self.ws:
            try:
                asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            except: pass
        while not self.send_queue.empty():
            try: self.send_queue.get_nowait()
            except queue.Empty: break
        if self._output_stream:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

# ==============================================================================
# Public API functions
# ==============================================================================

def init_session():
    global _session
    with _session_lock:
        if _session is None or not _session.running:
            _session = AISession()
            _session.start()
        return _session

def command(text: str) -> None:
    session = init_session()
    _chat_history.append(f"Пользователь: {text}")
    session.send_message({"text": text})

def send_frame() -> None:
    from r2_app.high_level import get_raw_frame
    session = init_session()
    frame = get_raw_frame(left=True)
    if frame is None:
        print("[send_frame] No frame captured")
        return
    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ret:
        print("[send_frame] JPEG encoding failed")
        return
    jpeg_bytes = jpeg.tobytes()
    _frame_buffer.append(jpeg_bytes)
    b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
    session.send_message({"image_b64": b64, "mime_type": "image/jpeg"})

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