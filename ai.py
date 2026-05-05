"""
AI Library for R2 Robot – Thin client using stateful proxy.
Command() sends text, send_frame() sends camera frame.
Answers are received asynchronously, audio is played, code executed.
"""

import asyncio
import base64
import io
import json
import re
import sys
import threading
import time
import cv2
import numpy as np
import sounddevice as sd
import websockets

# Конфигурация
PROXY_WS_URL = "wss://proxy-gemini-rlj1.onrender.com/ws/live"
HARDWARE_RATE = 48000
VOICE = "Enceladus"

_audio_enabled = True
_current_response = ""
_output_stream = None

# Объекты для фонового приёма ответов
_ws = None
_loop = None          # asyncio event loop в отдельном потоке
_receiver_thread = None
_running = False

OBSCURED_API_KEY = "c1RZQWNjZG8xT3ZHMV9IdldFVTMzakNfU3dhQ19PVWtEeVNheklB"

def get_real_key(obscured):
    try:
        decoded = base64.b64decode(obscured).decode()
        return decoded[::-1]
    except Exception:
        return None
API_KEY = get_real_key(OBSCURED_API_KEY)

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
3. После выполнения кода ты получишь сообщение "[SYSTEM_LOGS]" с результатами. Проанализируй их и дай ответ пользователю ОБЫЧНЫМ ТЕКСТОМ.
4. ТВОЙ ОТВЕТ БЕЗ БЛОКА #EXECUTE ЯВЛЯЕТСЯ СИГНАЛОМ ЗАВЕРШЕНИЯ ЗАДАЧИ.
5. Пиши код только если тебе реально нужно получить данные или выявить команду.
6. Не пиши ничего после блока кода. Закончился блок кода - ВСЁ. КОНЕЦ. Только после следующего сообщения можешь что-то писать.
"""

CONFIG = {
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

# -----------------------------------------------------------------------------
# Функции, доступные модели (для #EXECUTE)
# -----------------------------------------------------------------------------
def log(message: str) -> None:
    formatted_msg = f"[LOG]: {message}"
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

# -----------------------------------------------------------------------------
# Внутренний приёмник ответов (работает в фоновом asyncio‑потоке)
# -----------------------------------------------------------------------------
async def _receiver_loop():
    global _current_response
    while _running:
        try:
            raw = await _ws.recv()
            data = json.loads(raw)
        except Exception:
            print("Proxy connection lost, reconnecting...")
            await _connect_to_proxy()
            continue

        if "text" in data:
            _current_response = data["text"]
            print(f"\n🤖 Модель: {_current_response}")

            # Проверка на #EXECUTE
            code_match = re.search(r"#EXECUTE\n(.*?)\n#END", _current_response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                print(f"\n⚡ Выполняю код:\n{code}\n")

                # Перехват stdout
                captured_output = io.StringIO()
                exec_error = None
                original_stdout = sys.stdout
                sys.stdout = captured_output
                try:
                    exec(code, {"__builtins__": __builtins__}, _AI_EXEC_GLOBALS)
                except Exception as e:
                    exec_error = e
                finally:
                    sys.stdout = original_stdout

                output = captured_output.getvalue()
                captured_output.close()

                # Выводим захваченный вывод в консоль для контроля
                if output.strip():
                    print(f"📋 Логи выполнения:\n{output.strip()}")
                else:
                    print("📋 Логи выполнения: (пусто)")

                if exec_error:
                    print(f"❌ Ошибка выполнения: {exec_error}")
                    system_log_text = f"[SYSTEM_LOGS]\nОшибка: {exec_error}\n[END_LOGS]"
                else:
                    system_log_text = f"[SYSTEM_LOGS]\n{output.strip()}\n[END_LOGS]"

                # Отправляем модели
                if _ws is not None:
                    msg = {"text": system_log_text, "turn_complete": True}
                    try:
                        print(f"📤 Отправляю модели логи...")
                        await _ws.send(json.dumps(msg))
                        print(f"✅ Логи отправлены успешно")
                    except Exception as send_err:
                        print(f"❌ Не удалось отправить логи: {send_err}")
                else:
                    print("❌ WebSocket не подключён, не могу отправить логи")

        if "audio" in data and _audio_enabled:
            _play_audio(data["audio"])

        if data.get("end_of_turn"):
            _current_response = ""
            print("🔚 Конец ответа модели\n")

def _play_audio(audio_b64: str):
    global _output_stream
    try:
        missing = len(audio_b64) % 4
        if missing:
            audio_b64 += '=' * (4 - missing)
        raw_bytes = base64.b64decode(audio_b64)
        if len(raw_bytes) % 2:
            raw_bytes = raw_bytes[:-1]
        arr = np.frombuffer(raw_bytes, dtype=np.int16)
        if arr.size == 0:
            return
        if _output_stream is None:
            _output_stream = sd.OutputStream(
                samplerate=HARDWARE_RATE, channels=1, dtype='int16',
                latency='low', device=1

            )
            _output_stream.start()
        _output_stream.write(np.repeat(arr, 2))
    except Exception as e:
        print(f"Audio error: {e}")

async def _connect_to_proxy():
    global _ws
    while _running:
        try:
            _ws = await websockets.connect(
                PROXY_WS_URL,
                ping_interval=20,
                ping_timeout=20
            )
            setup_msg = {
                "api_key": API_KEY,
                "model_id": "gemini-3.1-flash-live-preview",
                "config": CONFIG
            }
            await _ws.send(json.dumps(setup_msg))
            print("✅ Подключено к прокси")
            return
        except Exception as e:
            print(f"⏳ Подключение к прокси не удалось: {e}. Повтор через 5 с...")
            await asyncio.sleep(5)

def _run_event_loop():
    global _loop, _running
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _loop = loop
    loop.run_until_complete(_connect_to_proxy())
    loop.run_until_complete(_receiver_loop())

# -----------------------------------------------------------------------------
# Публичные функции (вызываются из main.py)
# -----------------------------------------------------------------------------
def init():
    global _running, _receiver_thread
    if _running:
        return
    _running = True
    _receiver_thread = threading.Thread(target=_run_event_loop, daemon=True)
    _receiver_thread.start()

def command(text: str):
    """Отправить текстовую команду. Возвращает управление мгновенно."""
    if not _loop or not _ws:
        print("⚠️ Не подключено – нет соединения с прокси")
        return
    msg = {"text": text, "turn_complete": True}
    asyncio.run_coroutine_threadsafe(_ws.send(json.dumps(msg)), _loop)

def send_frame():
    """Отправить кадр с камеры."""
    if not _loop or not _ws:
        return
    from r2_app.high_level import get_raw_frame
    frame = get_raw_frame(left=True)
    if frame is None:
        return
    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ret:
        return
    b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    msg = {"image_b64": b64, "mime_type": "image/jpeg"}
    asyncio.run_coroutine_threadsafe(_ws.send(json.dumps(msg)), _loop)

def get_current_response():
    return _current_response

def enable_ai_audio(enabled: bool):
    global _audio_enabled
    _audio_enabled = enabled

def cleanup():
    global _running
    _running = False
    if _output_stream:
        _output_stream.stop()
        _output_stream.close()