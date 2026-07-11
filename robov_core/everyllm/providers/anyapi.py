from __future__ import annotations

from ..exceptions import ProviderError
from .base import BaseProvider


class AnyApiProvider(BaseProvider):
    name = "anyapi"
    needs_key = True
    key_name = "anyapi"
    _base_url = "https://api.anyapi.ai/v1"
    model_map = {
        "anyapi/openai/gpt-4o": "openai/gpt-4o",
        "anyapi/openai/gpt-4o-mini": "openai/gpt-4o-mini",
        "anyapi/deepseek/deepseek-r1": "deepseek/deepseek-r1",
        "anyapi/deepseek/deepseek-v3": "deepseek/deepseek-v3",
        "anyapi/meta-llama/llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
        "anyapi/auto": "anyapi/auto",
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
