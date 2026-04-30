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

# Параметры звука
AUDIO_INPUT = True  # Новый параметр для включения Live API
MODEL_RATE = 24000  # Выходная частота голоса Gemini
HARDWARE_RATE = 48000
INPUT_SAMPLERATE = 48000
SEND_SAMPLERATE = 16000
VOICE = "Enceladus"
MAX_HISTORY_CHARS = 8000

# Для совместимости со старым кодом
BASE_URL = "https://proxy-gemini-rlj1.onrender.com"
WS_URL = "wss://proxy-gemini-rlj1.onrender.com/ws/live"
MIC_DEVICE = 1

# Глобальные переменные состояния
is_ai_busy = False
audio_queue = asyncio.Queue()
_chat_history = []
_input_stream = None
_output_stream = None
last_audio_chunk = None  # Кэш для переотправки при ошибке

# Для совместимости со старым кодом
_audio_enabled = True
_initialized = False
_current_response = ""
_voice_mode_active = False
_ai_executing = False
_audio_input_active = False
_mic_queue = None
_mic_stream = None
_last_transcribed_text = ""
_execution_logs = []

BANNED_PHRASES = ["i'm"]

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

# --- High-level functions available to AI ---
def log(message: str) -> None:
    """Log a message (not visible to user)."""
    formatted_msg = f"[LOG]: {message}"
    _chat_history.append(formatted_msg)
    _execution_logs.append(formatted_msg)
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


def get_current_response():
    return _current_response


def enable_ai_audio(enabled: bool) -> None:
    """Enable or disable audio output from AI."""
    global _audio_enabled
    _audio_enabled = bool(enabled)
    print(f"[AI] Audio {'enabled' if _audio_enabled else 'disabled'}")


def build_system_prompt(chat_history):
    if not chat_history:
        return CHARACTER_PROMPT
    history_text = "\n".join(chat_history)[-MAX_HISTORY_CHARS:]
    return f"{CHARACTER_PROMPT}\n\nКОНТЕКСТ ДИАЛОГА ДО ПОСЛЕДНЕГО СБОЯ:\n{history_text}"


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


def audio_callback(indata, frames, time_info, status):
    """
    Колбэк микрофона. Срабатывает постоянно.
    Пишет звук в очередь только если ИИ не занят (не говорит и не выполняет код).
    """
    if status:
        print(f"Mic status: {status}")
    
    if not is_ai_busy:
        try:
            # indata.copy() важен, чтобы данные не затерлись в памяти SoundDevice
            audio_queue.put_nowait(indata.copy())
        except asyncio.QueueFull:
            pass


async def send_audio_loop(ws):
    """Асинхронная задача для ресэмплинга и отправки аудио-чанков."""
    global last_audio_chunk
    
    # Если остался недоотправленный чанк с прошлой сессии (переотправка)
    if last_audio_chunk:
        try:
            await ws.send(json.dumps(last_audio_chunk))
            last_audio_chunk = None
        except Exception:
            pass

    while True:
        indata = await audio_queue.get()
        if is_ai_busy:
            continue

        # Ресэмплинг 48000 -> 16000 (уменьшаем в 3 раза)
        resampled = _scipy_resample(indata[:, 0], len(indata) // 3)
        chunk_bytes = np.int16(resampled).tobytes()
        encoded = base64.b64encode(chunk_bytes).decode('utf-8')
        
        msg = {
            "realtimeInput": {
                "audio": {
                    "data": encoded,
                    "mimeType": f"audio/pcm;rate={SEND_SAMPLERATE}"
                }
            }
        }
        
        try:
            last_audio_chunk = msg  # Кэшируем перед отправкой
            await ws.send(json.dumps(msg))
            last_audio_chunk = None # Успешно отправлено, сбрасываем кэш
        except Exception as e:
            print(f"Ошибка отправки аудио, переподключение... ({e})")
            raise e # Прокидываем ошибку, чтобы сессия перезапустилась


async def handle_tool_call(ws, tool_call):
    """Обработка вызова функций (инструментов) моделью."""
    function_responses = []
    for fc in tool_call.get("functionCalls", []):
        func_name = fc.get('name')
        func_args = fc.get('args', {})
        print(f"Выполнение инструмента: {func_name} с аргументами {func_args}")
        
        # Выполняем соответствующую функцию
        result = {"status": "error", "message": "Unknown function"}
        if func_name == "log":
            log(func_args.get("message", ""))
            result = {"status": "success"}
        elif func_name == "set_emote":
            result = {"status": "success", "result": set_emote(func_args.get("emotion", ""))}
        elif func_name == "set_eyes":
            set_eyes(func_args.get("x", 0), func_args.get("y", 0))
            result = {"status": "success"}
        elif func_name == "get_cpu_temp":
            result = {"status": "success", "result": get_cpu_temp()}
        elif func_name == "get_system_stats":
            result = {"status": "success", "result": get_system_stats()}
        else:
            # Попробуем выполнить как Python код для #EXECUTE
            result = {"status": "success", "result": await execute_python(func_args.get("code", "") if func_args else "")}
        
        function_responses.append({
            "name": func_name,
            "id": fc.get("id"),
            "response": {"result": result}
        })

    tool_response_message = {
        "toolResponse": {
            "functionResponses": function_responses
        }
    }
    await ws.send(json.dumps(tool_response_message))


async def receive_loop(ws):
    """Асинхронная задача для приема данных от ИИ (текст, звук, инструменты)."""
    global is_ai_busy, _current_response, _chat_history
    assistant_text_accum = ""
    
    async for message in ws:
        try:
            resp = json.loads(message)
            
            # Обработка контента от сервера
            if "serverContent" in resp:
                sc = resp["serverContent"]

                # 1. Транскрипция (сохраняем ТОЛЬКО речь пользователя)
                if "inputTranscription" in sc:
                    text = sc["inputTranscription"].get("text", "").strip()
                    if text:
                        _chat_history.append(f"Пользователь: {text}")
                        print(f"Пользователь: {text}")

                # 2. Воспроизведение аудио от ИИ
                if "modelTurn" in sc:
                    is_ai_busy = True # ИИ начал говорить/отвечать
                    for part in sc["modelTurn"].get("parts", []):
                        if "inlineData" in part:
                            pcm_bytes = base64.b64decode(part["inlineData"]["data"])
                            if _output_stream:
                                _output_stream.write(np.frombuffer(pcm_bytes, dtype=np.int16))
                        # Обработка текста в modelTurn
                        if "text" in part:
                            chunk = part["text"]
                            assistant_text_accum += chunk
                            clean_chunk = re.sub(r"\+(?=[а-яёa-z])", "", chunk, flags=re.IGNORECASE)
                            if clean_chunk:
                                print(clean_chunk, end="", flush=True)
                                _current_response += clean_chunk

                # 3. ИИ закончил говорить
                if sc.get("turnComplete"):
                    is_ai_busy = False
                    if assistant_text_accum.strip():
                        cleaned = re.sub(r'\+', '', assistant_text_accum.strip())
                        _chat_history.append(f"Ассистент: {cleaned}")
                        print("\n")
                    _current_response = ""
                    assistant_text_accum = ""
            
            # Обработка вызова кода/инструментов
            if "toolCall" in resp:
                is_ai_busy = True # ИИ выполняет код, микрофон на паузе
                await handle_tool_call(ws, resp["toolCall"])
                is_ai_busy = False

        except Exception as e:
            print(f"Ошибка приема данных: {e}")
            raise e


async def start_voice_mode(initial_text: str = None):
    """Инициализация железа и запуск персистентной сессии WebSockets."""
    global _input_stream, _output_stream, is_ai_busy
    
    ws_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={API_KEY}"

    if not _output_stream:
        _output_stream = sd.OutputStream(samplerate=MODEL_RATE, channels=1, dtype='int16')
        _output_stream.start()

    if not _input_stream:
        # Вход: устройство 1, частота 48000
        _input_stream = sd.InputStream(
            device=MIC_DEVICE, 
            samplerate=INPUT_SAMPLERATE, 
            channels=1, 
            dtype='int16', 
            callback=audio_callback
        )
        _input_stream.start()

    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                print("\n[LIVE API] Успешно подключено!")
                is_ai_busy = False # Сброс состояния при новом коннекте
                
                # 1. Отправляем Setup-конфиг
                config_message = {
                    "setup": {
                        "model": f"models/{MODEL_ID}",
                        "systemInstruction": {
                            "parts": [{"text": build_system_prompt(_chat_history)}]
                        },
                        "generationConfig": {
                            "responseModalities": ["AUDIO"],
                            "speechConfig": {
                                "voiceConfig": {
                                    "prebuiltVoiceConfig": {
                                        "voiceName": VOICE
                                    }
                                }
                            }
                        }
                    }
                }
                await ws.send(json.dumps(config_message))

                # 2. Отправляем начальный текст (если это первый запуск из main.py)
                if initial_text:
                    await ws.send(json.dumps({"realtimeInput": {"text": initial_text}}))
                    initial_text = None

                # 3. Запускаем параллельные циклы приема и отправки
                send_task = asyncio.create_task(send_audio_loop(ws))
                recv_task = asyncio.create_task(receive_loop(ws))

                # Ждем, пока одна из задач не завершится (например, из-за ошибки)
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()

        except Exception as e:
            print(f"\n[LIVE API] Потеряно соединение: {e}. Переподключение через 2 секунд...")
            is_ai_busy = True # Блокируем микрофон на время реконнекта
            await asyncio.sleep(2)


# Оставил старый метод как фоллбэк на случай AUDIO_INPUT = False
async def _command_async_fallback(text: str) -> str:
    """Старый метод через proxy с поддержкой #EXECUTE."""
    global _output_stream, _chat_history, _current_response
    
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
            config = {
                "response_modalities": ["AUDIO"],
                "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": VOICE}}},
                "system_instruction": build_system_prompt(_chat_history)
            }
            await websocket.send(json.dumps({"api_key": API_KEY, "model_id": MODEL_ID, "config": config}))
            # Send user command
            await websocket.send(json.dumps({"text": text}))
            print(f"INPUT -> AI: {text}")
            _chat_history.append(f"Пользователь: {text}")

            # Get response with #EXECUTE support
            assistant_response = await _main_loop_with_execute(websocket)
            if assistant_response:
                _chat_history.append(f"Ассистент: {assistant_response}")

            print("\n")
            return assistant_response if assistant_response else "Нет ответа."

    except Exception as e:
        return f"Ошибка: {str(e)}"


async def _main_loop_with_execute(websocket):
    """Main interaction loop with #EXECUTE support."""
    global _current_response
    _current_response = ""
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

        if data.get("end_of_turn"):
            break

    # Check for #EXECUTE blocks
    code_match = re.search(r"#EXECUTE\n(.*?)\n#END", full_text_response, re.DOTALL)
    if code_match:
        report = await execute_python(code_match.group(1).strip())
        _chat_history.append(f"[SYSTEM_LOGS]: {report}")
        await websocket.send(json.dumps({"text": f"[SYSTEM_LOGS]:\n{report}"}))
        # Continue listening for response after execution
        return await _main_loop_with_execute(websocket)

    # Remove all '+' characters from final response
    cleaned_response = re.sub(r'\+', '', full_text_response.strip())
    return cleaned_response


def command(text: str) -> str:
    """
    Отправляет команду в ИИ. 
    Если AUDIO_INPUT активен, переводит приложение в бесконечный голосовой режим.
    """
    if AUDIO_INPUT:
        print("Запуск голосового режима (Live API)...")
        # asyncio.run заблокирует текущий поток, как и предполагается в main.py
        try:
            asyncio.run(start_voice_mode(text))
        except KeyboardInterrupt:
            cleanup()
        return ""
    else:
        return asyncio.run(_command_async_fallback(text))


def cleanup():
    """Очистка ресурсов перед выходом."""
    global _output_stream, _input_stream
    if _output_stream:
        _output_stream.stop()
        _output_stream.close()
        _output_stream = None
    if _input_stream:
        _input_stream.stop()
        _input_stream.close()
        _input_stream = None


if __name__ == "__main__":
    # Test the library
    print("Testing AI library...")
    response = command("Привет! Как дела?")
    print(f"Response: {response}")
    cleanup()