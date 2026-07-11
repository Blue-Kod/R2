from __future__ import annotations

from .base import (
    BaseProvider,
    http_request,
    http_stream_request,
    parse_chunk,
    parse_completion,
)
from ..types import ChatCompletion, ChatCompletionChunk


class OpenCodeProvider(BaseProvider):
    name = "opencode"
    needs_key = True
    key_name = "opencode"
    _base_url = "https://opencode.ai/zen/v1/chat/completions"
    supports_tools = True
    model_map = {
        "big-pickle": "big-pickle",
        "deepseek-v4-flash-free": "deepseek-v4-flash-free",
        "mimo-v2.5-free": "mimo-v2.5-free",
        "hy3-free": "hy3-free",
        # Too slow with big context   "nemotron-3-ultra-free": "nemotron-3-ultra-free",
        # Not so smart                "north-mini-code-free": "north-mini-code-free",
    }

    def __init__(self, api_key: str):
        self._api_key = api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _call(self, messages, model, max_tokens, stream, **kwargs):
        if stream:
            return self._stream(messages, model, max_tokens, **kwargs)
        return self._non_stream(messages, model, max_tokens, **kwargs)

    def _non_stream(self, messages, model, max_tokens, **kwargs):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        data = http_request(self._base_url, payload, self._headers())
        return parse_completion(data, model)

    def _stream(self, messages, model, max_tokens, **kwargs):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        for obj in http_stream_request(self._base_url, payload, self._headers()):
            chunk = parse_chunk(obj, model)
            if chunk is not None:
                yield chunk
