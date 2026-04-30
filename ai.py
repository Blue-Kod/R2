"""
AI Library for R2 Robot - High-level interface for AI interactions.
Provides command() to send text and receive text/audio response.
No microphone input.
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

def get_current_response():
    return _current_response


def enable_ai_audio(enabled: bool) -> None:
    """Enable or disable audio output from AI."""
    global _audio_enabled
    _audio_enabled = bool(enabled)
    print(f"[AI] Audio {'enabled' if _audio_enabled else 'disabled'}")


# --- High-level functions available to AI ---
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


_AI_EXEC_GLOBALS = {
    "log": log,
    "print": log,
    "set_emote": set_emote,
    "set_eyes": set_eyes,
    "get_cpu_temp": get_cpu_temp,
    "get_system_stats": get_system_stats,
}

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


# --- Text‑only response handling (no microphone) ---

async def receive_turn(websocket):
    """Receive one turn from the AI (text + optional audio)."""
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

    if is_refusal:
        print("[Система: Сообщение заблокировано фильтром отказов]")

    return full_text_response


async def main_loop(websocket):
    """Main interaction loop for one command. Handles code execution."""
    global _chat_history

    while True:
        full_text_response = await receive_turn(websocket)
        code_match = re.search(r"#EXECUTE\n(.*?)\n#END", full_text_response, re.DOTALL)

        if code_match:
            report = await execute_python(code_match.group(1).strip())
            _chat_history.append(f"[SYSTEM_LOGS]: {report}")
            await websocket.send(json.dumps({"text": f"[SYSTEM_LOGS]:\n{report}"}))
            continue

        # Remove all stress markers '+' from final response
        cleaned_response = re.sub(r'\+', '', full_text_response.strip())
        return cleaned_response


async def _command_async(text: str) -> str:
    """Async implementation of command function."""
    global _output_stream, _chat_history

    # Initialize audio output if needed
    if _output_stream is None and _audio_enabled:
        _output_stream = sd.OutputStream(
            samplerate=HARDWARE_RATE,
            channels=1,
            dtype='int16',
            device=1,
            latency='low'
        )
        _output_stream.start()

    # Wake up the proxy service
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

    try:
        async with websockets.connect(WS_URL, open_timeout=30) as websocket:
            config = build_config(_chat_history)
            await websocket.send(json.dumps({
                "api_key": API_KEY,
                "model_id": MODEL_ID,
                "config": config
            }))

            # Send user command as text
            await websocket.send(json.dumps({"text": text}))
            print(f"INPUT -> AI: {text}")
            _chat_history.append(f"Пользователь: {text}")

            # Get response (text + optional audio)
            assistant_response = await main_loop(websocket)
            if assistant_response:
                _chat_history.append(f"Ассистент: {assistant_response}")

            print("\n")
            return assistant_response if assistant_response else "Нет ответа."

    except Exception as e:
        return f"Ошибка: {str(e)}"


def command(text: str) -> str:
    """
    Send a text command to the AI and get a text response.
    Audio output is played automatically if enabled.
    """
    try:
        return asyncio.run(_command_async(text))
    except Exception as e:
        return f"Ошибка выполнения: {str(e)}"

def cleanup():
    global _output_stream
    if _output_stream:
        _output_stream.stop()
        _output_stream.close()
        _output_stream = None


if __name__ == "__main__":
    print("Testing AI library...")
    response = command("Привет! Как дела?")
    print(f"Response: {response}")
    cleanup()