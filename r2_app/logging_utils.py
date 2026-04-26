from collections import deque
from datetime import datetime
from threading import Lock
from typing import List


class AppLogger:
    def __init__(self, buffer_size: int = 500) -> None:
        self._buffer = deque(maxlen=buffer_size)
        self._lock = Lock()

    def log(self, *parts: object) -> str:
        message = " ".join(str(part) for part in parts)
        full_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(full_message)
        with self._lock:
            self._buffer.append(full_message)
        return full_message

    def recent(self) -> List[str]:
        with self._lock:
            return list(self._buffer)

