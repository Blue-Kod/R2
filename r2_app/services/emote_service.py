import threading


class EmoteService:
    def __init__(self) -> None:
        self._current = "normal"
        self._eyes_x = 0.0
        self._eyes_y = 0.0
        self._lock = threading.Lock()
        self._supported = {
            "normal",
            "sad",
            "excited",
            "spooked",
            "unamused",
            "worried",
            "woozy",
            "angry",
            "wince",
        }

    def set_emote(self, emotion_name: str) -> bool:
        name = str(emotion_name or "").strip().lower()
        if name not in self._supported:
            return False
        with self._lock:
            self._current = name
        return True

    def get_emote(self) -> str:
        with self._lock:
            return self._current

    def set_eyes_position(self, x: float, y: float) -> None:
        # x/y are normalized offsets from center, range: -1.0..1.0
        x = max(-1.0, min(1.0, float(x)))
        y = max(-1.0, min(1.0, float(y)))
        with self._lock:
            self._eyes_x = x
            self._eyes_y = y

    def get_eyes_position(self):
        with self._lock:
            return self._eyes_x, self._eyes_y

    def supported_emotes(self):
        return sorted(self._supported)

