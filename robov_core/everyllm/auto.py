from __future__ import annotations

from typing import Optional

VISION_TEST_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQU"
    "AAAADUExURf8AABniCTcAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAZdEVYdFNvZnR3YXJlAFBhaW50Lk5FVCA1"
    "LjEuMTITAUd0AAAAuGVYSWZJSSoACAAAAAUAGgEFAAEAAABKAAAAGwEFAAEAAABSAAAAKAEDAAEAAAACAAAA"
    "MQECABEAAABaAAAAaYcEAAEAAABsAAAAAAAAAGAAAAABAAAAYAAAAAEAAABQYWludC5ORVQgNS4xLjEzAAAD"
    "AACQBwAEAAAAMDIzMAGgAwABAAAAAQAAAAWgBAABAAAAlgAAAAAAAAACAAEAAgAEAAAAUjk4AAIABwAEAAAA"
    "MDEwMAAAAADZp5qVybcLXwAAAAxJREFUGNNjYBhcAAAAkAABcZnsWwAAAABJRU5ErkJggg=="
)

VISION_PROMPT = "What color is this image? Reply with RED or BLUE."

MODEL_CAPABILITIES: dict[str, dict] = {
    "glm-4.7-flash": {"vision": False, "thinking": True, "configurable_thinking": True},
    "agnes-2.0-flash": {"vision": True, "thinking": False, "configurable_thinking": False},
    "big-pickle": {"vision": False, "thinking": True, "configurable_thinking": False},
    "deepseek-v4-flash-free": {"vision": False, "thinking": True, "configurable_thinking": False},
    "mimo-v2.5-free": {"vision": False, "thinking": False, "configurable_thinking": False},
    "hy3-free": {"vision": False, "thinking": True, "configurable_thinking": False},
    "nemotron-3-ultra-free": {"vision": False, "thinking": True, "configurable_thinking": False},
    "north-mini-code-free": {"vision": False, "thinking": True, "configurable_thinking": False},
    "anyapi/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": {"vision": True, "thinking": True, "configurable_thinking": False},
    "anyapi/nvidia/nemotron-3-ultra-550b-a55b:free": {"vision": False, "thinking": True, "configurable_thinking": False},
    "anyapi/nvidia/nemotron-nano-9b-v2:free": {"vision": False, "thinking": False, "configurable_thinking": False},
    "anyapi/openai/gpt-4o": {"vision": True, "thinking": False, "configurable_thinking": False},
    "anyapi/openai/gpt-4o-mini": {"vision": False, "thinking": False, "configurable_thinking": False},
    "anyapi/deepseek/deepseek-r1": {"vision": False, "thinking": True, "configurable_thinking": False},
    "anyapi/deepseek/deepseek-v3": {"vision": False, "thinking": False, "configurable_thinking": False},
    "anyapi/meta-llama/llama-3.3-70b-instruct": {"vision": False, "thinking": False, "configurable_thinking": False},
    "anyapi/auto": {"vision": False, "thinking": False, "configurable_thinking": False},
}


def get_capabilities(model_name: str) -> dict:
    return MODEL_CAPABILITIES.get(model_name, {
        "vision": False, "thinking": False, "configurable_thinking": False,
    })


class TTFTTracker:
    def __init__(self):
        self._scores: dict[str, list[float]] = {}
        self._max_history = 5
        self._streaming_ok: dict[str, bool] = {}
        self._capabilities: dict[str, dict] = {}

    def update(self, model_name: str, ttft_ms: float):
        if model_name not in self._scores:
            self._scores[model_name] = []
        history = self._scores[model_name]
        history.append(ttft_ms)
        if len(history) > self._max_history:
            history.pop(0)

    def set_streaming(self, model_name: str, ok: bool):
        self._streaming_ok[model_name] = ok

    def supports_streaming(self, model_name: str) -> bool:
        return self._streaming_ok.get(model_name, True)

    def set_capabilities(self, model_name: str, vision: bool, thinking, configurable_thinking=False):
        self._capabilities[model_name] = {
            "vision": vision,
            "thinking": thinking,
            "configurable_thinking": configurable_thinking,
        }

    def get_capabilities(self, model_name: str) -> dict:
        if model_name in self._capabilities:
            return self._capabilities[model_name]
        return get_capabilities(model_name)

    def best_model(self) -> Optional[str]:
        avgs = self.get_scores()
        if not avgs:
            return None
        return min(avgs, key=avgs.get)

    def get_scores(self) -> dict[str, float]:
        return {
            m: round(sum(ts) / len(ts), 1)
            for m, ts in self._scores.items()
            if ts
        }

    def has_data(self) -> bool:
        return bool(self._scores)


class AutoRouter:
    def __init__(self, model_registry: dict[str, object], tracker: TTFTTracker):
        self._registry = model_registry
        self._tracker = tracker

    def select(self) -> tuple[str, object]:
        best = self._tracker.best_model()
        if best and best in self._registry:
            return best, self._registry[best]

        for name, provider in self._registry.items():
            return name, provider

        from .exceptions import EveryLLMError

        raise EveryLLMError("No providers available for auto model")
