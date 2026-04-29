"""
AI Library for R2 Robot - High-level interface for AI interactions.
Provides command(), enable_ai_audio(), and start_voice_mode() functions.
"""

import asyncio
import json
import base64
import re
import traceback
import time
import requests
import numpy as np
import sounddevice as sd
import websockets
import threading
from collections import deque
from scipy.signal import resample as _scipy_resample

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
INPUT_SAMPLERATE = 44100
TARGET_SAMPLERATE = 16000         # ожидаемый Gemini Live
MIC_DEVICE = 1                    # индекс микрофона

# Global state
_audio_enabled = True
_chat_history = []
_output_stream = None
_initialized = False
_current_response = ""
_voice_mode_active = False
_ai_executing = False
_audio_input_active = False
_mic_queue = None
_mic_stream = None
_last_transcribed_text = ""       # последняя реплика пользователя

def get_current_response():
    return _current_response


def enable_ai_audio(enabled: bool) -> None:
    """Enable or disable audio output from AI."""
    global _audio_enabled
    _audio_enabled = bool(enabled)
    print(f"[AI] Audio {'enabled' if _audio_enabled else 'disabled'}")


# --- High-level functions available to AI ---
# These are the only functions the AI can execute via #EXECUTE blocks

def log(message: str) -> None:
    """Log a message (not visible to user)."""
    formatted_msg = f"[LOG]: {message}"
    _chat_history.append(formatted_msg)
    print(formatted_msg)


def set_emote(emotion: str) -> bool:
    """Set robot emote. Returns success."""
    from r2_app.high_level import emote
    print(f"[AI-EXEC] Setting emote: {emotion}")
    return emote(emotion)


def set_eyes(x: float, y: float) -> None:
    """Set eyes position (-1.0 to 1.0)."""
    from r2_app.high_level import set_eyes_position
    print(f"[AI-EXEC] Setting eyes: x={x}, y={y}")
    set_eyes_position(x, y)


def get_cpu_temp() -> str:
    """Get CPU temperature."""
    from r2_app.high_level import cpu_temp
    temp = cpu_temp()
    log(temp)
    return temp


def get_system_stats() -> dict:
    """Get system statistics."""
    from r2_app.high_level import health_snapshot
    stats = health_snapshot()
    log(stats)
    return stats


# Available functions for AI execution
_AI_EXEC_GLOBALS = {
    "log": log,
    "print": log,
    "set_emote": set_emote,
    "set_eyes": set_eyes,
    "get_cpu_temp": get_cpu_temp,
    "get_system_stats": get_system_stats,
}

# Execution logs
_execution_logs = []

def _log(message):
    formatted_msg = f"[LOG]: {message}"
    _execution_logs.append(formatted_msg)
    print(formatted_msg)


# --- Character Prompt ---
CHARACTER_PROMPT = """
Ты — робот R2. Стиль: краткий, немного официальный.
Говори на русском. Всегда используй '+' перед ударными гласными. Мужской род.
Говор+и внятно, не тор+опся. Произнос+и слов+а полн+остью, избег+ай сокращ+ений. Твоя речь должна быть разборчивой, как у диктора.

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
    if not chat_history:
        return CHARACTER_PROMPT
    history_text = "\n".join(chat_history)[-MAX_HISTORY_CHARS:]
    return f"{CHARACTER_PROMPT}\n\nКОНТЕКСТ ДИАЛОГА ДО ПОСЛЕДНЕГО СБОЯ:\n{history_text}"


def build_config(chat_history):
    config = dict(CONFIG_BASE)
    config["system_instruction"] = build_system_prompt(chat_history)
    return config


async def execute_python(code):
    """Execute Python code with limited high-level functions."""
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


# --- Аудио‑захват (микрофон) ---

def _resample_to_16000(audio_chunk: np.ndarray, orig_rate: int) -> np.ndarray:
    """Переводит из orig_rate в 16 кГц."""
    if orig_rate == 16000:
        return audio_chunk
    duration = len(audio_chunk) / orig_rate
    target_samples = int(duration * 16000)
    return _scipy_resample(audio_chunk, target_samples).astype(np.int16)


def _mic_callback_factory(loop: asyncio.AbstractEventLoop, audio_queue: asyncio.Queue):
    """Создаёт callback для sounddevice.InputStream, помещающий аудио в очередь."""
    def callback(indata, frames, time, status):
        if status:
            print(f"[Mic] status: {status}")
        # indata: (frames, channels) float32
        mono = indata[:, 0].copy()
        # Преобразуем в int16
        int16_data = (mono * 32767).astype(np.int16)
        # Ресемплинг
        resampled = _resample_to_16000(int16_data, INPUT_SAMPLERATE)
        # В байты
        audio_bytes = resampled.tobytes()
        asyncio.run_coroutine_threadsafe(audio_queue.put(audio_bytes), loop)
    return callback


def _start_microphone(loop: asyncio.AbstractEventLoop, audio_queue: asyncio.Queue):
    """Запуск микрофона в отдельном потоке."""
    global _mic_stream
    if _mic_stream is not None:
        return
    try:
        _mic_stream = sd.InputStream(
            samplerate=INPUT_SAMPLERATE,
            device=MIC_DEVICE,
            channels=1,
            dtype='float32',
            blocksize=1024,
            callback=_mic_callback_factory(loop, audio_queue)
        )
        _mic_stream.start()
        print("[AI] Microphone started")
    except Exception as e:
        print(f"[AI] Failed to start microphone: {e}")


def _stop_microphone():
    global _mic_stream
    if _mic_stream is not None:
        _mic_stream.stop()
        _mic_stream.close()
        _mic_stream = None
        print("[AI] Microphone stopped")


async def _audio_sender_task(websocket, audio_queue: asyncio.Queue):
    """Читает аудио из очереди и отправляет на WebSocket, пока разрешено."""
    global _audio_input_active, _ai_executing
    while True:
        chunk = await audio_queue.get()
        if _audio_input_active and not _ai_executing:
            audio_msg = {
                "audio": {
                    "data": base64.b64encode(chunk).decode('utf-8'),
                    "mimeType": "audio/pcm;rate=16000"
                }
            }
            try:
                await websocket.send(json.dumps(audio_msg))
            except Exception as e:
                print(f"[AI] Audio send error: {e}")
                break
        # иначе отбрасываем (не слушаем/выполняется код)


# --- Голосовой диалог ---

async def _voice_interaction_loop(websocket):
    """Непрерывный цикл: слушаем пользователя, обрабатываем ответы ИИ."""
    global _audio_input_active, _ai_executing, _last_transcribed_text, _chat_history, _current_response
    assistant_text_accum = ""
    is_refusal = False

    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue(maxsize=500)
    _start_microphone(loop, audio_queue)
    sender_task = asyncio.create_task(_audio_sender_task(websocket, audio_queue))
    _audio_input_active = True

    try:
        while True:
            try:
                raw_data = await asyncio.wait_for(websocket.recv(), timeout=30)
                data = json.loads(raw_data)
            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError:
                continue

            # Транскрипция речи пользователя
            if "input_text" in data and data["input_text"]:
                user_speech = data["input_text"].strip()
                if user_speech:
                    _last_transcribed_text = user_speech
                    _chat_history.append(f"Пользователь: {user_speech}")
                    print(f"[AI] User said: {user_speech}")

            # Текст ассистента
            if "text" in data:
                chunk = data["text"]
                assistant_text_accum += chunk
                clean_chunk = re.sub(r"\+(?=[а-яёa-z])", "", chunk, flags=re.IGNORECASE)
                if clean_chunk:
                    print(clean_chunk, end="", flush=True)
                    _current_response += clean_chunk

            if any(phrase in _current_response.lower() for phrase in BANNED_PHRASES):
                is_refusal = True
                is_refusal = True

            # Аудио‑ответ (воспроизведение)
            if "audio" in data and not is_refusal and _audio_enabled:
                text_so_far = assistant_text_accum.lower()
                is_code_block = "#execute" in text_so_far and "#end" not in text_so_far
                if not is_code_block:
                    try:
                        audio_b64 = data["audio"]
                        missing_padding = len(audio_b64) % 4
                        if missing_padding:
                            audio_b64 += '=' * (4 - missing_padding)
                        raw_bytes = base64.b64decode(audio_b64)
                        if len(raw_bytes) % 2 != 0:
                            raw_bytes = raw_bytes[:-1]
                        audio_array = np.frombuffer(raw_bytes, dtype=np.int16)
                        if audio_array.size > 0:
                            mono_48k = np.repeat(audio_array, 2)
                            if _output_stream:
                                _output_stream.write(mono_48k)
                    except Exception as e:
                        print(f"\n[Ошибка аудио]: {e}")

            # Проверка на #EXECUTE
            code_match = re.search(r"#EXECUTE\n(.*?)\n#END", assistant_text_accum, re.DOTALL)
            if code_match:
                _ai_executing = True
                report = await execute_python(code_match.group(1).strip())
                _chat_history.append(f"[SYSTEM_LOGS]: {report}")
                await websocket.send(json.dumps({"text": f"[SYSTEM_LOGS]:\n{report}"}))
                assistant_text_accum = assistant_text_accum.replace(code_match.group(0), "")
                _ai_executing = False
                continue

            # Конец реплики
            if data.get("end_of_turn"):
                if assistant_text_accum.strip():
                    cleaned = re.sub(r'\+', '', assistant_text_accum.strip())
                    _chat_history.append(f"Ассистент: {cleaned}")
                    print("\n")
                # Сброс накопленного
                _current_response = ""
                assistant_text_accum = ""

    except websockets.exceptions.ConnectionClosed:
        print("[AI] Voice WebSocket closed")
    except Exception as e:
        print(f"[AI] Voice error: {e}")
    finally:
        _audio_input_active = False
        _ai_executing = False
        _stop_microphone()
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass


def start_voice_mode():
    """
    Запускает непрерывный голосовой диалог.
    Блокирует выполнение до завершения сессии.
    """
    global _voice_mode_active
    if _voice_mode_active:
        print("[AI] Voice mode already active")
        return

    async def _run():
        global _voice_mode_active, _output_stream
        _voice_mode_active = True

        # Инициализация аудиовыхода, если ещё нет
        if _output_stream is None and _audio_enabled:
            _output_stream = sd.OutputStream(
                samplerate=HARDWARE_RATE,
                channels=1,
                dtype='int16',
                device=1,
                latency='low'
            )
            _output_stream.start()

        # Подогрев сервиса
        for attempt in range(1, 6):
            try:
                response = requests.get(BASE_URL, timeout=10)
                if response.status_code < 500:
                    break
            except:
                pass
            await asyncio.sleep(2)

        try:
            async with websockets.connect(WS_URL, open_timeout=30) as websocket:
                config = build_config(_chat_history)
                await websocket.send(json.dumps({
                    "api_key": API_KEY,
                    "model_id": MODEL_ID,
                    "config": config
                }))
                await _voice_interaction_loop(websocket)
        except Exception as e:
            print(f"[AI] Voice mode error: {e}")
            # При ошибке переподключаемся и отправляем последнюю фразу текстом
            if _last_transcribed_text:
                print(f"[AI] Re-sending as text: {_last_transcribed_text}")
                try:
                    command(_last_transcribed_text)
                except:
                    pass
        finally:
            _voice_mode_active = False

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


# --- Текстовый режим command (без изменений, как в оригинале) ---

async def receive_turn(websocket, full_text_response_buffer):
    global _current_response
    _current_response = ""
    """Receive one turn from the AI."""
    full_text_response = ""
    is_refusal = False
    print("AI: ", end="", flush=True)

    while True:
        try:
            raw_data = await asyncio.wait_for(websocket.recv(), timeout=30)
            data = json.loads(raw_data)
        except asyncio.TimeoutError:
            print("\n[Система: Тайм-аут ответа модели]")
            break
        except json.JSONDecodeError:
            print("\n[Система: Получен некорректный JSON от сервера]")
            continue

        if "text" in data:
            chunk = data["text"]
            full_text_response += chunk

            if any(phrase in full_text_response.lower() for phrase in BANNED_PHRASES):
                is_refusal = True

            if not is_refusal:
                display_text = re.sub(r"\+(?=[а-яёa-z])", "", chunk, flags=re.IGNORECASE)
                if display_text:
                    print(display_text, end="", flush=True)
                    _current_response += display_text

        if "audio" in data and not is_refusal and _audio_enabled:
            text_so_far = full_text_response.lower()
            is_code_block = "#execute" in text_so_far and "#end" not in text_so_far

            if not is_code_block:
                try:
                    # 1. Декодируем и чиним паддинг
                    audio_b64 = data["audio"]
                    missing_padding = len(audio_b64) % 4
                    if missing_padding:
                        audio_b64 += '=' * (4 - missing_padding)

                    raw_bytes = base64.b64decode(audio_b64)
                    if len(raw_bytes) % 2 != 0:
                        raw_bytes = raw_bytes[:-1]

                    audio_array = np.frombuffer(raw_bytes, dtype=np.int16)

                    if audio_array.size > 0:
                        # 2. Ресемплинг 24к -> 48к
                        mono_48k = np.repeat(audio_array, 2)

                        if _output_stream:
                            # Теперь stereo_audio имеет форму (кол-во семплов, 2)
                            _output_stream.write(mono_48k)

                except Exception as e:
                    print(f"\n[Ошибка аудио]: {e}")

        if data.get("end_of_turn"):
            break

    if is_refusal:
        print("[Система: Сообщение заблокировано фильтром отказов]")

    return full_text_response


async def main_loop(websocket):
    """Main interaction loop for one command."""
    global _chat_history

    while True:
        full_text_response = await receive_turn(websocket, "")
        code_match = re.search(r"#EXECUTE\n(.*?)\n#END", full_text_response, re.DOTALL)

        if code_match:
            report = await execute_python(code_match.group(1).strip())
            _chat_history.append(f"[SYSTEM_LOGS]: {report}")
            await websocket.send(json.dumps({"text": f"[SYSTEM_LOGS]:\n{report}"}))
            continue

        # --- Убираем все символы '+' из финального ответа ---
        cleaned_response = re.sub(r'\+', '', full_text_response.strip())
        return cleaned_response


async def _command_async(text: str) -> str:
    """Async implementation of command function."""
    global _output_stream, _chat_history
    # Initialize audio stream if needed
    if _output_stream is None and _audio_enabled:
        _output_stream = sd.OutputStream(
            samplerate=HARDWARE_RATE,
            channels=1,
            dtype='int16',
            device=1,
            latency='low'
        )
        _output_stream.start()
    # Wake up the service
    for attempt in range(1, 6):
        try:
            response = requests.get(BASE_URL, timeout=10)
            if response.status_code < 500:
                print(f"[Система: Render прогрет ({response.status_code})]")
                break
        except requests.RequestException:
            pass
        if attempt < 5:
            time.sleep(2)

    # Connect to WebSocket
    try:
        async with websockets.connect(WS_URL, open_timeout=30) as websocket:
            config = build_config(_chat_history)
            await websocket.send(json.dumps({"api_key": API_KEY, "model_id": MODEL_ID, "config": config}))
            # Send user command
            await websocket.send(json.dumps({"text": text}))
            print(f"INPUT -> AI: {text}")
            _chat_history.append(f"Пользователь: {text}")

            # Get response
            assistant_response = await main_loop(websocket)
            if assistant_response:
                _chat_history.append(f"Ассистент: {assistant_response}")

            print("\n")
            return assistant_response if assistant_response else "Нет ответа."

    except Exception as e:
        return f"Ошибка: {str(e)}"


def command(text: str) -> str:
    """
    Send a command to the AI and get response.
    Args:
        text: Command text to send to AI
    Returns:
        AI response as string (without '+' characters)
    """
    try:
        return asyncio.run(_command_async(text))
    except Exception as e:
        return f"Ошибка выполнения: {str(e)}"


def cleanup():
    """Clean up resources."""
    global _output_stream
    if _output_stream:
        _output_stream.stop()
        _output_stream.close()
        _output_stream = None


if __name__ == "__main__":
    # Test the library
    print("Testing AI library...")
    response = command("Привет! Как дела?")
    print(f"Response: {response}")
    cleanup()