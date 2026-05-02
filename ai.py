"""
AI Library for R2 Robot - Enhanced Reconnection & Memory Logic.
Features: session resumption, client content injection, local frame cache,
thread-safe message queue (uses standard queue.Queue to avoid event loop issues).
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
MODEL_RATE = 24000
HARDWARE_RATE = 48000
BASE_URL = "https://proxy-gemini-rlj1.onrender.com"
WS_URL = "wss://proxy-gemini-rlj1.onrender.com/ws/live"
VOICE = "Enceladus"
MAX_HISTORY_CHARS = 8000
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BASE_DELAY = 2.0  # seconds
FRAME_HISTORY_LENGTH = 15    # Keep last 15 frames

# Global state
_audio_enabled = True
_chat_history = []  # List of strings
_output_stream = None
_current_response = ""
_session: Optional['AISession'] = None
_session_lock = threading.Lock()
_frame_buffer: deque[bytes] = deque(maxlen=FRAME_HISTORY_LENGTH)

# ==============================================================================
# Helper Functions & Character Prompt
# ==============================================================================

def get_current_response():
    return _current_response

def enable_ai_audio(enabled: bool) -> None:
    global _audio_enabled
    _audio_enabled = bool(enabled)
    print(f"[AI] Audio {'enabled' if _audio_enabled else 'disabled'}")

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
... (твой существующий промпт) ...
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
        # Thread-safe queue for messages from other threads
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
                # If we get here without exception, session ended normally
                break
            except (websockets.ConnectionClosed, asyncio.TimeoutError, OSError) as e:
                reconnect_attempt += 1
                print(f"Connection lost (attempt {reconnect_attempt}/{MAX_RECONNECT_ATTEMPTS}): {e}")
                delay = RECONNECT_BASE_DELAY * reconnect_attempt
                time.sleep(delay)  # blocking sleep in this thread is fine
            except Exception as e:
                print(f"Unexpected error in session: {e}")
                break

        self.running = False

    async def _connect_and_process(self):
        config = build_config(_chat_history)
        setup_msg = {"api_key": API_KEY, "model_id": MODEL_ID, "config": config}
        if self._session_token:
            setup_msg["session_resumption"] = {"handle": self._session_token}

        async with websockets.connect(WS_URL, open_timeout=30) as ws:
            self.ws = ws
            await ws.send(json.dumps(setup_msg))
            print("WebSocket session opened")

            # Inject local history if we have frames
            if _frame_buffer:
                await self._send_history_context(ws)

            # Start parallel tasks
            receiver_task = asyncio.create_task(self._receiver())
            sender_task = asyncio.create_task(self._sender())

            # Wait for either task to complete
            done, pending = await asyncio.wait(
                [receiver_task, sender_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            # Re-raise exception if any
            for task in done:
                if task.exception():
                    raise task.exception()

    async def _send_history_context(self, ws):
        """Inject client content (text history + recent frames)."""
        global _frame_buffer, _chat_history

        # Build turns from chat history
        turns = []
        for msg in _chat_history:
            role = "user" if msg.startswith("Пользователь:") else "model"
            turns.append({"role": role, "parts": [{"text": msg}]})

        # Build image parts from frame buffer
        image_parts = []
        for jpeg_bytes in _frame_buffer:
            b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            image_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

        client_content_msg = {
            "clientContent": {
                "turns": turns,
                "turn_complete": True,
                "extra_parts": image_parts
            }
        }
        await ws.send(json.dumps(client_content_msg))
        print("Injected local history into session.")

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

            # Extract session token if provided
            if "sessionResumption" in data:
                self._session_token = data["sessionResumption"]["handle"]
                print("Session resumption token stored.")

            if "text" in data:
                clean = re.sub(r"\+", "", data["text"])
                _current_response += clean
                print(clean, end="", flush=True)

            if "audio" in data and _audio_enabled:
                try:
                    audio_b64 = data["audio"]
                    # fix padding
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
                            # upsample to 48kHz
                            mono_48k = np.repeat(audio_array, 2)
                            self._output_stream.write(mono_48k)
                except Exception as e:
                    print(f"\nAudio error: {e}")

            if data.get("end_of_turn"):
                print()  # newline after turn
                _current_response = ""

    async def _sender(self):
        """Regularly check the thread-safe queue and send any messages."""
        while True:
            try:
                msg = self.send_queue.get_nowait()
                try:
                    await self.ws.send(json.dumps(msg))
                except Exception as e:
                    print(f"Send error: {e}")
            except queue.Empty:
                pass
            await asyncio.sleep(0.05)  # prevent busy loop

    def start(self):
        """Start the AISession in a background thread."""
        if self.running:
            return
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
        """Thread-safe message insertion. Can be called from any thread."""
        if self.running:
            self.send_queue.put(msg)

    def stop(self):
        """Request session stop and clean up."""
        self.running = False
        # Schedule connection close if still alive
        if self.loop and self.ws:
            try:
                asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            except Exception:
                pass
        # Clear the queue to unblock sender
        while not self.send_queue.empty():
            try:
                self.send_queue.get_nowait()
            except queue.Empty:
                break
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
    """Capture a frame, cache it, and send to AI session."""
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