import threading
from typing import Optional


class EmoteService:
    def __init__(self, eye_api=None) -> None:
        self._current = "normal"
        self._eyes_x = 0.0
        self._eyes_y = 0.0
        self._lock = threading.Lock()
        self._eye_api = eye_api  # Reference to EyeAPI for pushing updates to display
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

    def set_eye_api(self, eye_api) -> None:
        """Set the EyeAPI reference for pushing updates to the pywebview display."""
        self._eye_api = eye_api

    def set_emote(self, emotion_name: str) -> bool:
        name = str(emotion_name or "").strip().lower()
        if name not in self._supported:
            return False
        with self._lock:
            self._current = name
        # Push to pywebview display if available
        if self._eye_api:
            try:
                self._eye_api.update_emote(name)
            except Exception as e:
                # Display might not be ready yet, log and continue
                import logging
                logging.getLogger(__name__).debug(f"Failed to push emote to display: {e}")
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
        # Push to pywebview display if available
        if self._eye_api:
            try:
                self._eye_api.update_eyes_position(x, y)
            except Exception as e:
                # Display might not be ready yet, log and continue
                import logging
                logging.getLogger(__name__).debug(f"Failed to push eyes position to display: {e}")

    def get_eyes_position(self):
        with self._lock:
            return self._eyes_x, self._eyes_y

    def supported_emotes(self):
        return sorted(self._supported)

