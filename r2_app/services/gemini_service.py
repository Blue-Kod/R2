from dataclasses import dataclass
from typing import Optional


@dataclass
class GeminiResult:
    ok: bool
    text: str
    error: Optional[str] = None


class GeminiGateway:
    """
    High-level adapter for future Gemini API integration.
    The application works even when Gemini is not configured.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-pro") -> None:
        self._api_key = api_key
        self._model = model

    def is_ready(self) -> bool:
        return bool(self._api_key)

    def test(self, prompt: str = "Ping from R2") -> GeminiResult:
        if not self._api_key:
            return GeminiResult(
                ok=False,
                text="",
                error="Gemini API key is not configured",
            )

        # Placeholder for actual SDK call. Current contract keeps main.py stable.
        return GeminiResult(
            ok=True,
            text=f"Gemini gateway is ready for model {self._model}. Prompt: {prompt}",
            error=None,
        )

