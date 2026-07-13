from __future__ import annotations

from ..exceptions import ProviderError
from .base import BaseProvider


class AnyApiProvider(BaseProvider):
    name = "anyapi"
    needs_key = True
    key_name = "anyapi"
    _base_url = "https://api.anyapi.ai/v1"
    model_map = {
        "anyapi/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "anyapi/nvidia/nemotron-3-ultra-550b-a55b:free": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "anyapi/nvidia/nemotron-nano-9b-v2:free": "nvidia/nemotron-nano-9b-v2:free"
    }

    def __init__(self, api_key: str):
        self._api_key = api_key

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError(
                self.name,
                "openai package not installed. Run: pip install openai",
            )

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
        )
